/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_MRLDA_H
#define SIGTM_MRLDA_H

#if SIG_MSVC_ENV

#include "common/lda_module.hpp"
#include "../util/mapreduce_module.h"
#include "../../external/mapreduce/include/mapreduce.hpp"
#include "../../external/boost_sub/math/special_functions/polygamma.hpp"
#include "boost/math/special_functions/digamma.hpp"
#include "SigUtil/lib/functional/fold.hpp"

namespace sigtm
{
struct mrlda::MapValue;

template<class T> using MatrixDV = VectorD<VectorV<T>>;		// document - word

using  boost::math::digamma;
using  boost::math::trigamma;

template <class C>
double calcModule0(C const& vec)
{
	const auto vec_sum = sig::sum(vec);
	return lgamma(vec_sum)
		- sig::sum(vec, [&](double v){ return lgamma(v); })
		+ sig::sum(vec, [&](double v){ return (v - 1) * (digamma(v) - digamma(vec_sum)); });
}


/// Latent Dirichlet Allocation on mapreduce (estimate by Variational Bayesian inference)
/**
*/
class MrLDA final : public LDA, private impl::LDA_Module, public std::enable_shared_from_this<MrLDA>
{
	static const double global_convergence_threshold;
	//static const double local_convergence_threshold;
	static const uint max_local_iteration_;

	friend class sigtm::mrlda::datasource::MRInputIterator;

public:
	enum class ReduceKeyType{ Lambda, Alpha, Liklihood };

	using map_key_type = DocumentId;
	using map_value_type = mrlda::MapValue;
	using reduce_key_type = std::tuple<
		ReduceKeyType,
		typename std::common_type<uint, TopicId>::type,
		typename std::common_type<TopicId, WordId>::type>;
	using reduce_value_type = double;	// sufficient statistics
	using mr_input_iterator = mrlda::datasource::MRInputIterator;
	using mr_performance_result = mapreduce::results;

private:
	class MapTask : public mapreduce::map_task<map_key_type, map_value_type>
	{
		void process(value_type const& value, VectorK<double>& gamma, MatrixVK<double>& omega) const;
			
	public:
		template<typename Runtime>
		void operator()(Runtime& runobj, key_type const& key, value_type& value) const
		{
			VectorK<double>& gamma = *value.gamma_;
			MatrixVK<double> omega(value.vnum_, std::vector<double>(value.knum_, 0));
			double lhm1 = 0, lhm2 = 0;

			process(value, gamma, omega);

			const double dig_sum_gamma = digamma(sig::sum(gamma));
			for (uint k = 0; k < value.knum_; ++k){
				const double ddg = digamma(gamma[k]) - dig_sum_gamma;
				const double sum_log_beta = sig::foldl([](double s, double b){ return s + std::log(b); }, 0, (*value.phi_)[k]);

				for (uint v = 0; v < value.vnum_; ++v){
					if ((*value.word_ct_)[v] != 0){
						const double wp = (*value.word_ct_)[v] * omega[v][k];
						runobj.emit_intermediate(std::make_tuple(ReduceKeyType::Lambda, k, v), wp);
						lhm1 += (wp * ddg);
						lhm2 += (wp * (sum_log_beta - std::log(omega[v][k])));
						//std::cout << omega[v][k] << ", " << sum_log_beta << ", " << std::log(omega[v][k]) << std::endl;
					}
				}
				runobj.emit_intermediate(std::make_tuple(ReduceKeyType::Alpha, zero, k), ddg);
				//std::cout << k << "ddg:" << ddg << std::endl;
			}
			runobj.emit_intermediate(std::make_tuple(ReduceKeyType::Liklihood, zero, zero), lhm1 + lhm2 - calcModule0(gamma));
			std::cout << lhm1 << ", " << lhm2 << ", " << calcModule0(gamma) << std::endl;
		}
	};
	
	class ReduceTask : public mapreduce::reduce_task<reduce_key_type, reduce_value_type>
	{
	public:
		template<typename Runtime, typename Iter>
		void operator()(Runtime& runobj, key_type const& key, Iter it, Iter const end) const
		{
			runobj.emit(key, std::accumulate(it, end, 0.0));
		}
	};
	
	using mr_job = mapreduce::job<
		sigtm::MrLDA::MapTask,
		sigtm::MrLDA::ReduceTask,
		mapreduce::null_combiner,
		mr_input_iterator
	>;

private:
	DocumentSetPtr input_data_;
	MatrixDV<uint> doc_word_ct_;	// word frequency in each document
	
	const uint D_;		// number of documents
	const uint K_;		// number of topics
	const uint V_;		// number of words
		
	VectorK<double> alpha_;			// dirichlet hyper parameter of theta
	MatrixKV<double> eta_;			// dirichlet hyper parameter of phi
	MatrixKV<double> phi_;			// parameter of word distribution
	
	MatrixDK<double> gamma_;		// variational parameter of theta(document-topic)
	//MatrixKV<double> lambda_;		// variational parameter of phi(topic-word)
	
	std::unique_ptr<mr_job> mapreduce_;
	mrlda::Specification mr_spec_;
	mr_performance_result performance_result_;
	
	MatrixKV<double> term_score_;	// word score of emphasizing each topic
	uint total_iter_ct_;
	const double term3_;		// use for calculate liklihood

	sig::SimpleRandom<double> rand_d_;

private:
	MrLDA() = delete;
	MrLDA(MrLDA const&) = delete;
	MrLDA(MrLDA&&) = delete;

	MrLDA(bool resume, uint num_topics, DocumentSetPtr input_data, Maybe<VectorK<double>> alpha, Maybe<VectorV<double>> beta, mrlda::Specification spec) :
		input_data_(input_data), D_(input_data->getDocNum()), K_(num_topics), V_(input_data->getWordNum()),
		alpha_(isJust(alpha) ? fromJust(alpha) : SIG_INIT_VECTOR(double, K, default_alpha_base / K_)), eta_(MatrixKV<double>(K_, isJust(beta) ? fromJust(beta) : SIG_INIT_VECTOR(double, V, default_beta))),
		mapreduce_(nullptr), mr_spec_(spec), total_iter_ct_(0), term3_(sig::sum(eta_, [&](VectorV<double> const& v){ return calcModule0(v); })), rand_d_(0.0, 1.0, FixedRandom)
	{
		init(resume); 
	}

	void init(bool resume);

	void initMR(){
		mrlda::datasource::MRInputIterator::reset();
		//mapreduce_ = std::make_unique<mr_job>(mr_input_iterator(shared_from_this(), mr_spec_), mr_spec_); 
		//performance_result_ = mr_performance_result();
	}

	void saveResumeData() const;
	
	// 収束判定に使う尤度の計算
	double calcLiklihood(double term2, double term4) const;

	// mapreduceの1サイクルの処理
	// return -> 尤度
	double iteration();

public:
	~MrLDA() = default;
	
	DynamicType getDynamicType() const override{ return DynamicType::MRLDA; }

	template <Distribution Select>
	auto compare(Id id1, Id id2) const->typename Map2Cmp<Select>::type { return compareDefault<Select>(id1, id2, D_, K_); }

public:
	// 以下ユーザインタフェース

	/**
	\brief
		@~japanese ファクトリ関数 (デフォルト設定で使用する場合)	\n
		@~english factory function (construct model with default settings)	\n

	\details
		@~japanese
		ハイパーパラメータは[α = 50/num_topics, β = 0.01]に設定．\n
		ハイパーパラメータに関する詳細は \ref g_hparam_setting を参照．

		\param num_topics トピック数（潜在因子の次元数）
		\param input_data 文書データ（DocumentLoader および その派生クラスにより作成）
		\param resume 前回の学習途中のパラメータから学習を再開するか

		@~english
		Hyper-parameters are set as default [α = 50/num_topics, β = 0.01]. \n
		See \ref g_hparam_setting about details of hyper-parameters.

		\param num_topics number of topics（dimension of latent factor）
		\param input_data documents（instance of DocumentLoader or its derived class）
		\param resume whether reload previous trained parameters or not
	*/
	static LDAPtr makeInstance(
		uint num_topics,
		DocumentSetPtr input_data,
		bool resume = false
	){
		auto obj = std::shared_ptr<MrLDA>(new MrLDA(
			resume,
			num_topics,
			input_data,
			Nothing<VectorK<double>>(),
			Nothing<VectorV<double>>(),
			mrlda::Specification(impl::cpu_core_num, impl::cpu_core_num)
		));
		obj->mapreduce_ = std::make_unique<mr_job>(mr_input_iterator(obj, obj->mr_spec_), obj->mr_spec_);
		return obj;
	}
	
	/**
	\brief
		@~japanese ファクトリ関数 (α, β をsymmetricに設定して使用する場合)	\n
		@~english factory function (construct model with symmetric hyper-parameters)	\n

	\details
		@~japanese
		ハイパーパラメータ（ベクトル）の各要素をすべて同じ値に設定．\n
		デフォルト値を使う場合は sig::nothing または sig::Nothing<double>() を引数に指定．\n
		ハイパーパラメータに関する詳細は \ref sigtm_guide を参照．

		\param num_topics トピック数（潜在因子の次元数）
		\param input_data 文書データ（DocumentLoader および その派生クラスにより作成）
		\param alpha ディリクレ分布ハイパーパラメータα
		\param beta ディリクレ分布ハイパーパラメータβ
		\param resume 前回の学習途中のパラメータから学習を再開するか

		@~english
		In hyper-parameter vector, each element is set as the same value.	\n
		If want to set as default, pass either sig::nothing or sig::Nothing<double>() to the corresponding argument.	\n
		See \ref g_hparam_setting about details of hyper-parameters.

		\param num_topics number of topics（dimension of latent factor）
		\param input_data documents（instance of DocumentLoader or its derived class）
		\param alpha　hyper-parameter α
		\param beta hyper-parameter β
		\param resume whether reload previous trained parameters or not
	*/
	static LDAPtr makeInstance(
		uint num_topics,
		DocumentSetPtr input_data,
		Maybe<double> alpha,
		Maybe<double> beta = Nothing<double>(),
		bool resume = false
	){
		auto obj = std::shared_ptr<MrLDA>(new MrLDA(
			resume,
			num_topics,
			input_data,
			isJust(alpha) ? Just(VectorK<double>(num_topics, fromJust(alpha))) : Nothing<VectorK<double>>(),
			isJust(beta) ? Just(VectorV<double>(input_data->getWordNum(), fromJust(beta))) : Nothing<VectorV<double>>(),
			mrlda::Specification(impl::cpu_core_num, impl::cpu_core_num)
		));
		obj->mapreduce_ = std::make_unique<mr_job>(mr_input_iterator(obj, obj->mr_spec_), obj->mr_spec_);
		return obj;
	}
	
	/** 
	\brief
		@~japanese ファクトリ関数 (α, β をunsymmetricに設定して使用する場合)	\n
		@~english factory function (construct model with unsymmetric hyper-parameters)	\n

	\details
		@~japanese
		ハイパーパラメータのベクトルをすべて任意の値に設定．	\n
		デフォルト値を使う場合は sig::nothing または sig::Nothing<std::vector<double>>() を引数に指定．\n
		ハイパーパラメータに関する詳細は \ref sigtm_guide を参照．

		\param num_topics トピック数（潜在因子の次元数）
		\param input_data 文書データ（DocumentLoader および その派生クラスにより作成）
		\param alpha ディリクレ分布ハイパーパラメータα
		\param beta ディリクレ分布ハイパーパラメータβ
		\param resume 前回の学習途中のパラメータから学習を再開するか

		@~english
		In hyper-parameter vector, each element is set as arbitrary value.	\n
		If want to set as default, pass either sig::nothing or sig::Nothing<double>() to the corresponding argument.	\n
		See \ref g_hparam_setting about details of hyper-parameters.

		\param num_topics number of topics（dimension of latent factor）
		\param input_data documents（instance of DocumentLoader or its derived class）
		\param alpha　hyper-parameter α
		\param beta hyper-parameter β
		\param resume whether reload previous trained parameters or not
	*/
	static LDAPtr makeInstance(
		uint num_topics,
		DocumentSetPtr input_data, 
		Maybe<VectorK<double>> alpha,
		Maybe<VectorV<double>> beta = Nothing<VectorV<double>>(),
		bool resume = false
	){
		auto obj = std::shared_ptr<MrLDA>(new MrLDA(
			resume,
			num_topics,
			input_data,
			alpha,
			beta,
			mrlda::Specification(impl::cpu_core_num, impl::cpu_core_num)
		));
		obj->mapreduce_ = std::make_unique<mr_job>(mr_input_iterator(obj, obj->mr_spec_), obj->mr_spec_);
		return obj;
	}


	/** 
	\brief
		@~japanese モデルの学習を行う
		@~english Train model
	*/
	void train(uint num_iteration) override{ train(num_iteration, null_lda_callback); }

	/** 
	\brief
		@~japanese モデルの学習を行う
		@~english Train model
	*/
	void train(uint num_iteration, std::function<void(LDA const*)> callback) override;
	

	/**
	\brief 
		@~japanese コンソールに出力
		@~english Output to console
	*/
	void print(Distribution target) const override{ save(target, L""); }

	/**
	\brief 
		@~japanese ファイルに出力
		@~english Output to file
	*/
	void save(Distribution target, FilepassString save_dir, bool detail = false) const override;

	auto getWordOfDocument(uint num_get_words, DocumentId d_id) const->std::vector<std::tuple<std::wstring, double>>;

	/**
	\brief
		@~japanese 文書が持つトピック比率を取得
		@~english Get the topic proportion of each document
	*/
	auto getTheta() const->MatrixDK<double> override{ return LDA::getTheta(); }

	/**
	\brief
		@~japanese 指定文書が持つトピック比率を取得
		@~english Get the topic proportion of specified document
	*/
	auto getTheta(DocumentId d_id) const->VectorK<double> override;

	
	/** 
	\brief
		@~japanese トピックが持つ語彙比率を取得
		@~english Get the word proportion of each topic
	*/
	auto getPhi() const->MatrixKV<double> override{ return LDA::getPhi(); }

	/**
	\brief
		@~japanese 指定トピックが持つ語彙比率を取得
		@~english Get the word proportion of specified topic
	*/
	auto getPhi(TopicId k_id) const->VectorV<double> override;
	

	/**
	\brief
		@~japanese トピックを強調する語彙スコアを取得
		@~english Get the word-scores of each topic
	*/
	auto getTermScore() const->MatrixKV<double> override{ return term_score_; }
	
	/**
	\brief
		@~japanese 指定トピックを強調する語彙スコアを取得
		@~english Get the word-scores of specified topic
	*/
	auto getTermScore(TopicId k_id) const->VectorV<double> override{ return term_score_[k_id]; }

	/**
	\brief
		@~japanese トピックの代表語彙とそのスコアを取得
		@~english Get the characteristic words and their scores of each topic
	*/
	auto getWordOfTopic(Distribution target, uint num_get_words) const->VectorK< std::vector< std::tuple<std::wstring, double> > > override{ return LDA::getWordOfTopic(target, num_get_words); }
	
	/**
	\brief
		@~japanese 指定トピックの代表語彙とそのスコアを取得
		@~english Get the characteristic words and their scores of specific topic
	*/
	auto getWordOfTopic(Distribution target, uint num_get_words, TopicId k_id) const->std::vector< std::tuple<std::wstring, double> > override;

	/**
	\brief
		@~japanese 文書の代表語彙とそのスコアを取得
		@~english Get the characteristic words and their scores of each document
	*/
	auto getWordOfDocument(Distribution target, uint num_get_words) const->VectorD< std::vector< std::tuple<std::wstring, double> > > override{ return LDA::getWordOfDocument(target, num_get_words); }
	
	/**
	\brief
		@~japanese 指定文書の代表語彙とそのスコアを取得
		@~english Get the characteristic words and their scores of specified document
	*/
	auto getWordOfDocument(Distribution target, uint num_get_words, DocumentId d_id) const->std::vector< std::tuple<std::wstring, double> > override;
	
	/**
	\brief
		@~japanese 文書数を取得
		@~english Get the number of documents
	*/
	uint getDocumentNum() const override{ return D_; }

	/**
	\brief
		@~japanese トピック数を取得
		@~english Get the number of topics
	*/
	uint getTopicNum() const override{ return K_; }

	/**
	\brief
		@~japanese 語彙数を取得
		@~english Get the number of words (vocabularies)
	*/
	uint getWordNum() const override{ return V_; }
	
	/**
	\brief
		@~japanese ハイパーパラメータαを取得（\ref g_hparam_alpha ）
		@~english Get \ref g_hparam_alpha
	*/
	auto getAlpha() const->VectorK<double> override{ return alpha_; }

	/**
	\brief
		@~japanese ハイパーパラメータβを取得（\ref g_hparam_beta ）
		@~english Get \ref g_hparam_beta
	*/
	auto getBeta() const->VectorV<double> override{
		VectorV<double> result;
		for(uint v=0; v<V_; ++v) result.push_back(sig::sum_col(eta_, v) / K_);	//average over documents
		return result;
	}
	
	/**
	\brief
		@~japanese モデルの対数尤度（\ref g_log_likelihood ）を取得
		@~english Get model \ref g_log_likelihood
	*/
	double getLogLikelihood() const override{ return calcLogLikelihood(input_data_->tokens_, getTheta(), getPhi()); }

	/**
	\brief
		@~japanese モデルの \ref g_perplexity を取得
		@~english Get model \ref g_perplexity
	*/
	double getPerplexity() const override{ return std::exp(-getLogLikelihood() / input_data_->tokens_.size()); }
};

}

namespace std
{
template <> struct hash<typename sigtm::MrLDA::reduce_key_type>
{
	size_t operator()(sigtm::MrLDA::reduce_key_type const& x) const
	{
		return hash<int>()(static_cast<int>(std::get<0>(x)))
			^ hash<typename std::common_type<sigtm::uint, sigtm::TopicId>::type>()(std::get<1>(x))
			^ hash<typename std::common_type<sigtm::TopicId, sigtm::WordId>::type>()(std::get<2>(x));			
	}
};
}	//std
#endif
#endif
