/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_MRLDA_H
#define SIGTM_MRLDA_H

#include "lda_common_module.hpp"
#include "../helper/mapreduce_module.h"
#include "../../external/mapreduce/include/mapreduce.hpp"
#include "../../external/boost_sub/math/special_functions/polygamma.hpp"
#include "boost/math/special_functions/digamma.hpp"
#include "SigUtil/lib/functional/fold.hpp"

#if USE_SIGNLP
#include "../helper/input_text.h"
#endif

namespace sigtm
{
class mrlda::MapValue;

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


/* Latent Dirichlet Allocation on mapreduce (estimate by Variational Bayesian inference) */
class MrLDA : public LDA, private impl::LDA_Module, public std::enable_shared_from_this<MrLDA>
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
	InputDataPtr input_data_;
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

	MrLDA(bool resume, uint topic_num, InputDataPtr input_data, Maybe<VectorK<double>> alpha, Maybe<VectorV<double>> beta, mrlda::Specification spec) :
		input_data_(input_data), D_(input_data->getDocNum()), K_(topic_num), V_(input_data->getWordNum()),
		alpha_(alpha ? sig::fromJust(alpha) : SIG_INIT_VECTOR(double, K, default_alpha_base / K_)), eta_(MatrixKV<double>(K_, beta ? sig::fromJust(beta) : SIG_INIT_VECTOR(double, V, default_beta))),
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
	~MrLDA(){}
	
	DynamicType getDynamicType() const override{ return DynamicType::MRLDA; }

	/* InputDataで作成した入力データを元にコンストラクト */
	// デフォルト設定で使用する場合
	static LDAPtr makeInstance(bool resume, uint topic_num, InputDataPtr input_data){
		auto obj = std::shared_ptr<MrLDA>(new MrLDA(resume, topic_num, input_data, nothing, nothing, mrlda::Specification(ThreadNum, ThreadNum)));
		obj->mapreduce_ = std::make_unique<mr_job>(mr_input_iterator(obj, obj->mr_spec_), obj->mr_spec_);
		return obj;
	}
	// alpha, beta をsymmetricに設定する場合
	static LDAPtr makeInstance(bool resume, uint topic_num, InputDataPtr input_data, double alpha, Maybe<double> beta = nothing){
		auto obj = std::shared_ptr<MrLDA>(new MrLDA(resume, topic_num, input_data, VectorK<double>(topic_num, alpha), beta ? sig::Just<VectorV<double>>(VectorV<double>(input_data->getWordNum(), sig::fromJust(beta))) : nothing, mrlda::Specification(ThreadNum, ThreadNum)));
		obj->mapreduce_ = std::make_unique<mr_job>(mr_input_iterator(obj, obj->mr_spec_), obj->mr_spec_);
		return obj;
	}
	// alpha, beta を多次元で設定する場合
	static LDAPtr makeInstance(bool resume, uint topic_num, InputDataPtr input_data, VectorK<double> alpha, Maybe<VectorV<double>> beta = nothing){
		auto obj = std::shared_ptr<MrLDA>(new MrLDA(resume, topic_num, input_data, alpha, beta, mrlda::Specification(ThreadNum, ThreadNum)));
		obj->mapreduce_ = std::make_unique<mr_job>(mr_input_iterator(obj, obj->mr_spec_), obj->mr_spec_);
		return obj;
	}

	/* モデルの学習を行う */
	// iteration_num: 学習の反復回数(mapreduce処理とその結果の統合で1反復とする)
	void train(uint iteration_num) override{ train(iteration_num, null_lda_callback); }

	// call_back: 毎回の反復終了時に行う処理
	void train(uint iteration_num, std::function<void(LDA const*)> callback) override;
	

	// 確率分布同士の類似度を測る(メソッドチェーンな感じに使用)
	// Select: LDA::Distributionから選択
	// id1,id2: 類似度を測る対象のindex
	// return -> 比較関数の選択(関数オブジェクト)
	template <Distribution Select>
	auto compare(Id id1, Id id2) const->typename Map2Cmp<Select>::type{ return compareDefault<Select>(id1, id2, D_, K_); }

	// コンソールに出力
	void print(Distribution target) const override{ save(target, L""); }

	// ファイルに出力
	void save(Distribution target, FilepassString save_folder, bool detail = false) const override;

	//ドキュメントのトピック分布 [doc][topic]
	auto getTheta() const->MatrixDK<double> override;
	auto getTheta(DocumentId d_id) const->VectorK<double> override;

	//トピックの単語分布 [topic][word]
	auto getPhi() const->MatrixKV<double> override;
	auto getPhi(TopicId k_id) const->VectorV<double> override;
	
	//トピックを強調する単語スコア [topic][word]
	auto getTermScore() const->MatrixKV<double> override{ return term_score_; }
	auto getTermScore(TopicId t_id) const->VectorV<double> override{ return term_score_[t_id]; }

	// 指定トピックの上位return_word_num個の、語彙とスコアを返す
	// [topic][ranking]<vocab, score>
	auto getWordOfTopic(Distribution target, uint return_word_num) const->VectorK< std::vector< std::tuple<std::wstring, double> > > override;
	// [ranking]<vocab, score>
	auto getWordOfTopic(Distribution target, uint return_word_num, TopicId k_id) const->std::vector< std::tuple<std::wstring, double> > override;

	// 指定ドキュメントの上位return_word_num個の、語彙とスコアを返す
	// [doc][ranking]<vocab, score>
	auto getWordOfDocument(uint return_word_num) const->VectorD< std::vector< std::tuple<std::wstring, double> > > override;
	//[ranking]<vocab, score>
	auto getWordOfDocument(uint return_word_num, DocumentId d_id) const->std::vector< std::tuple<std::wstring, double> > override;
	
	uint getDocumentNum() const override{ return D_; }
	uint getTopicNum() const override{ return K_; }
	uint getWordNum() const override{ return V_; }
	
	// get hyper-parameter of topic distribution
	auto getAlpha() const->VectorK<double> override{ return alpha_; }

	// get hyper-parameter of word distribution
	auto getBeta() const->VectorV<double> override{
		VectorV<double> result;
		for(uint v=0; v<V_; ++v) result.push_back(sig::sum_col(eta_, v) / K_);	//average over documents
		return result;
	}

	auto getGamma(DocumentId d_id) const->VectorK<double>{ return gamma_[d_id]; }
 
	//auto getLambda(TopicId k_id) const->VectorV<double>{ return lambda_[k_id]; }

	double getLogLikelihood() const override{ return calcLogLikelihood(input_data_->tokens_, getTheta(), getPhi()); }

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