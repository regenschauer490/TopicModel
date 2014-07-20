/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_MRLDA_H
#define  SIGTM_MRLDA_H

#include "lda_interface.hpp"
#include "../helper/mapreduce_module.h"
#include "../helper/input.h"
#include "../../external/mapreduce/include/mapreduce.hpp"
#include "boost/math/special_functions/digamma.hpp"

#if USE_SIGNLP
#include "../helper/input_text.h"
#endif

namespace sigtm
{
class mrlda::MapValue;

template<class T> using MatrixDV = VectorD<VectorV<T>>;		// document - word


using  boost::math::digamma;

inline double calcModule1(VectorK<double> const& gamma, uint k)
{
	return digamma(gamma[k]) - digamma(sig::sum(gamma));	//todo:第2項を呼び出し元でキャッシュ化する
}

/*inline double calcModule2(MatrixKV<double> const& lambda, uint k)
{
	const double sum = sig::sum_row(lambda, k);
	return std::accumulate(std::begin(lambda[k]), std::end(lambda[k]), 0.0, [sum](double s, double l){ return s + std::log(l / sum); });
}*/
inline double calcModule2(MatrixKV<double> const& beta, uint k)
{
	return std::accumulate(std::begin(beta[k]), std::end(beta[k]), 0.0, [](double s, double b){ return s + std::log(b); });
}

template <class C>
double calcModule0(C const& vec)
{
	const auto vec_sum = sig::sum(vec);
	return lgamma(vec_sum)
		- sig::sum(vec, [&](double v){ return lgamma(v); })
		+ sig::sum(vec, [&](double v){ return (v - 1) * (digamma(v) - digamma(vec_sum)); });
}


class MrLDA : public LDA, public std::enable_shared_from_this<MrLDA>
{
	static const double local_convergence;
	static const double global_convergence;

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
		void process(value_type const& value, VectorK<double>& gamma, MatrixVK<double>& phi) const;
			
	public:
		template<typename Runtime>
		void operator()(Runtime& runobj, key_type const& key, value_type& value) const
		{
			VectorK<double>& gamma = *value.gamma_;
			MatrixVK<double> phi(value.vnum_, std::vector<double>(value.knum_, 0));
			double lhm1 = 0, lhm2 = 0;

			process(value, gamma, phi);

			for (uint k = 0; k < value.knum_; ++k){
				const double ddg = calcModule1(gamma, k);
				const double sum_log_beta = calcModule2(*value.beta_, k);

				for (uint v = 0; v < value.vnum_; ++v){
					const double wp = (*value.word_ct_)[v] * phi[v][k];
					runobj.emit_intermediate(std::make_tuple(ReduceKeyType::Lambda, k, v), wp);
					lhm1 += (wp * ddg);
					lhm2 += (wp * (sum_log_beta - std::log(phi[v][k])));;		// todo:word_ct == 0 の場合の処理を分けて最適化
					//std::cout << phi[v][k] << ", " << sum_log_beta << ", " << std::log(phi[v][k]) << std::endl;
				}
				runobj.emit_intermediate(std::make_tuple(ReduceKeyType::Alpha, zero, k), ddg);
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
	
	class DriverTask
	{
	public:

	};

	using mr_job = mapreduce::job<
		sigtm::MrLDA::MapTask,
		sigtm::MrLDA::ReduceTask,
		mapreduce::null_combiner,
		mr_input_iterator
	>;

private:
	InputDataPtr input_data_;

	const uint D_;		// number of documents
	const uint K_;		// number of topics
	const uint V_;		// number of words
		
	VectorK<double> alpha_;			// dirichlet hyper parameter of gamma
	MatrixKV<double> eta_;			// dirichlet hyper parameter of lambda
	MatrixDK<double> gamma_;		// variational parameter of theta(document-topic)
	//MatrixKV<double> lambda_;		// variational parameter of beta(topic-word)
	MatrixKV<double> beta_;
	MatrixDV<uint> doc_word_ct_;	// word frequency in each document
	
	std::unique_ptr<mr_job> mapreduce_;
	mrlda::Specification mr_spec_;
	mr_performance_result performance_result_;
	
	sig::SimpleRandom<double> rand_d_;

	MatrixKV<double> term_score_;
	const double term3_;		// use for calculate liklihood

private:
	MrLDA() = delete;
	MrLDA(MrLDA const&) = delete;
	MrLDA(MrLDA&&) = delete;
	MrLDA(uint topic_num, InputDataPtr input_data, maybe<VectorK<double>> alpha, maybe<MatrixKV<double>> eta, mrlda::Specification spec) :
		input_data_(input_data), D_(input_data->doc_num_), K_(topic_num), V_(input_data->words_.size()),
		alpha_(alpha ? sig::fromJust(alpha) : VectorK<double>(K_, 50.0 / K_)), eta_(eta ? sig::fromJust(eta) : MatrixKV<double>(K_, VectorV<double>(V_, 0.1))),
		mapreduce_(nullptr), mr_spec_(spec), rand_d_(0.0, 1.0, FixedRandom), term3_(sig::sum(eta_, [&](VectorV<double> const& v){ return calcModule0(v); }))
	{
		init(); 
	}

	void init();

	void initMR(){
		mrlda::datasource::MRInputIterator::reset();
		//mapreduce_ = std::make_unique<mr_job>(mr_input_iterator(shared_from_this(), mr_spec_), mr_spec_); 
		//performance_result_ = mr_performance_result();
	}

	// 収束判定に使う尤度の計算
	double calcLiklihood(double term2, double term4) const;

	// mapreduceの1サイクルの処理
	// return -> 尤度
	double iteration();

public:
	~MrLDA(){}
	
	DynamicType getDynamicType() const override{ return DynamicType::MRLDA; }

	// InputDataで作成した入力データでコンストラクト
	static LDAPtr makeInstance(uint topic_num, InputDataPtr input_data, maybe<VectorK<double>> alpha = nothing, maybe<MatrixKV<double>> eta = nothing){
		auto obj = std::shared_ptr<MrLDA>(new MrLDA(topic_num, input_data, alpha, eta, mrlda::Specification(ThreadNum, ThreadNum)));
		obj->mapreduce_ = std::make_unique<mr_job>(mr_input_iterator(obj, obj->mr_spec_), obj->mr_spec_);
		return obj;
	}

	// 内部状態を更新する(学習)
	// 各ノードでの変分パラメータ更新の反復回数を指定
	void learn(uint iteration_num) override;

/*	// 確率分布同士の類似度を測る(メソッドチェーンな感じに使用)
	// Select: LDA::Distributionから選択, id1,id2：類似度を測る対象のindex
	// return -> 比較関数の選択(関数オブジェクト)
	template <Distribution Select>
	auto compare(Id id1, Id id2) const->typename Map2Cmp<Select>::type
	{}
*/

	// コンソールに出力
	void print(Distribution target) const override{ save(target, L""); }

	// ファイルに出力
	void save(Distribution target, FilepassString save_folder, bool detail = false) const override;

	//ドキュメントのトピック分布 [doc][topic]
	auto getTopicDistribution() const->MatrixDK<double> override;
	auto getTopicDistribution(DocumentId d_id) const->VectorK<double> override;

	//トピックの単語分布 [topic][word]
	auto getWordDistribution() const->MatrixKV<double> override;
	auto getWordDistribution(TopicId k_id) const->VectorV<double> override;
	
	//トピックを強調する語スコア [topic][word]
	auto getTermScoreOfTopic() const->MatrixKV<double> override;
	auto getTermScoreOfTopic(TopicId t_id) const->VectorV<double> override;

	//ドキュメントのThetaとTermScoreの積 [ranking]<word_id,score>
	auto getTermScoreOfDocument(DocumentId d_id) const->std::vector< std::tuple<WordId, double> > override;

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
	auto getEta() const->VectorV<double> override{
		VectorV<double> result;
		for(uint v=0; v<V_; ++v) result.push_back(sig::sum_col(eta_, v) / K_);	//average over documents
		return result; 
	}

	auto getGamma(DocumentId d_id) const->VectorK<double>{ return gamma_[d_id]; }
 
	//auto getLambda(TopicId k_id) const->VectorV<double>{ return lambda_[k_id]; }
};

}
#endif