﻿/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_LDA_GIBBS_H
#define SIGTM_LDA_GIBBS_H

#include "lda_interface.hpp"
#include "../helper/input.h"

#if USE_SIGNLP
#include "../helper/input_text.h"
#endif

#include "SigUtil/lib/tool.hpp"

namespace sigtm
{
/* Latent Dirichlet Allocation (estimate by Gibbs Sampling or Collapsed Gibbs Sampling) */
class LDA_Gibbs : public LDA
{
	InputDataPtr input_data_;
	TokenList const& tokens_;
	
	const uint D_;		// number of documents
	const uint K_;		// number of topics
	const uint V_;		// number of words

	VectorK<double> alpha_;			// dirichlet hyper parameter of theta
	VectorV<double> beta_;			// dirichlet hyper parameter of phi
	VectorT<uint> z_;				// topic assigned to each tokens temporary
			
	MatrixVK<uint> word_ct_;		// topic count of each words
	MatrixDK<uint> doc_ct_;			// topic count of each documents
	VectorK<uint> topic_ct_;		// topic count of all tokens

	double alpha_sum_;
	double beta_sum_;
	VectorK<double> tmp_p_;
	MatrixKV<double> term_score_;	// word score of emphasizing each topic
	uint total_iter_ct_;
	
	const std::function<double(LDA_Gibbs const* obj, Token const& t, uint k)> sampling_;
	sig::SimpleRandom<uint> rand_ui_;
	sig::SimpleRandom<double> rand_d_;

public:
	struct GibbsSampling
	{
		double operator()(LDA_Gibbs const* obj, Token const& t, uint k)
		{
			static int pre_k = -1;
			static double dk_sum = 0;
			double const& alpha = obj->alpha_[k];
			double const& beta = obj->beta_[t.word_id];
			
			if(k != pre_k) dk_sum = sig::sum(obj->doc_ct_[t.doc_id]);
			pre_k = k;
			return ((obj->doc_ct_[t.doc_id][k] + alpha) / (dk_sum + obj->alpha_sum_)) * ((obj->word_ct_[t.word_id][k] + beta) / (obj->topic_ct_[k] + obj->beta_sum_));
		}
	};

	struct CollapsedGibbsSampling
	{
		double operator()(LDA_Gibbs const* obj, Token const& t, uint k){
			double const& beta = obj->beta_[t.word_id];
			return (obj->doc_ct_[t.doc_id][k] + obj->alpha_[k]) * (obj->word_ct_[t.word_id][k] + beta) / (obj->topic_ct_[k] + obj->beta_sum_);
		}
	};

private:
	LDA_Gibbs() = delete;
	LDA_Gibbs(LDA_Gibbs const&) = delete;

	// alpha_:各単語のトピック更新時の選択確率平滑化定数, beta_:
	template <class SamplingMethod>
	LDA_Gibbs(SamplingMethod sm,bool resume,  uint topic_num, InputDataPtr input_data, maybe<VectorK<double>> alpha, maybe<VectorV<double>> beta) :
		D_(input_data->getDocNum()), K_(topic_num), V_(input_data->getWordNum()), input_data_(input_data),
		alpha_(alpha ? sig::fromJust(alpha) : VectorK<double>(K_, default_alpha_base / K_)), beta_(beta ? sig::fromJust(beta) : VectorV<double>(V_, default_beta)),
		tokens_(input_data->tokens_), word_ct_(V_, VectorK<uint>(K_, 0)), doc_ct_(D_, VectorK<uint>(K_, 0)), topic_ct_(K_, 0),
		alpha_sum_(0), beta_sum_(0), tmp_p_(K_, 0.0), z_(tokens_.size(), 0), term_score_(K_, VectorV<double>(V_, 0)), total_iter_ct_(0), 
		sampling_(SamplingMethod()), rand_ui_(0, K_ - 1, FixedRandom), rand_d_(0.0, 1.0, FixedRandom)
	{
		init(resume);
	}

	void init(bool resume);
	void update(Token const& t);
	void saveResumeData() const;

public:
	~LDA_Gibbs(){}

	DynamicType getDynamicType() const override{ return DynamicType::GIBBS; }

	/* InputDataで作成した入力データを元にコンストラクト */
	// デフォルト設定で使用する場合
	template <class SamplingMethod = CollapsedGibbsSampling>
	static LDAPtr makeInstance(bool resume, uint topic_num, InputDataPtr input_data){
		return LDAPtr(new LDA_Gibbs(SamplingMethod(), resume, topic_num, input_data, nothing, nothing));
	}
	// alpha, beta をsymmetricに設定する場合
	template <class SamplingMethod = CollapsedGibbsSampling>
	static LDAPtr makeInstance(bool resume, uint topic_num, InputDataPtr input_data, double alpha, maybe<double> beta = nothing){
		return LDAPtr(new LDA_Gibbs(SamplingMethod(), resume, topic_num, input_data, VectorK<double>(topic_num, alpha), beta ? sig::Just<VectorV<double>>(VectorV<double>(input_data->getWordNum(), sig::fromJust(beta))) : nothing));
	}
	// alpha, beta を多次元で設定する場合
	template <class SamplingMethod = CollapsedGibbsSampling>
	static LDAPtr makeInstance(bool resume, uint topic_num, InputDataPtr input_data, VectorK<double> alpha, maybe<VectorV<double>> beta = nothing){
		return LDAPtr(new LDA_Gibbs(SamplingMethod(), resume, topic_num, input_data, alpha, beta));
	}
	
	// モデルの学習を行う
	// iteration_num: 学習の反復回数(ギブスサンプリングによる全変数の更新を1反復とする)
	void train(uint iteration_num) override{ train(iteration_num, null_lda_callback);  }

	// call_back: 毎回の反復終了時に行う処理
	void train(uint iteration_num, std::function<void(LDA const*)> callback) override;


	// 確率分布同士の類似度を測る(メソッドチェーンな感じに使用)
	// Select: LDA::Distributionから選択
	// id1,id2: 類似度を測る対象のindex
	// return -> 比較関数の選択(関数オブジェクト)
	template <Distribution Select>
	auto compare(Id id1, Id id2) const->typename Map2Cmp<Select>::type{	return compareDefault<Select>(id1, id2, D_, K_); }

	// コンソールに出力
	void print(Distribution target) const override{ save(target, L""); }

	// ファイルに出力
	void save(Distribution target, FilepassString save_folder, bool detail = false) const override;

	//ドキュメントのトピック分布 [doc][topic]
	using LDA::getTheta;//auto getTheta() const->MatrixDK<double> override{ return LDA::getTheta(); }
	auto getTheta(DocumentId d_id) const->VectorK<double> override;

	//トピックの単語分布 [topic][word]
	auto getPhi() const->MatrixKV<double> override{ return LDA::getPhi(); }
	auto getPhi(TopicId k_id) const->VectorV<double> override;

	//トピックを強調する単語スコア [topic][word]
	auto getTermScore() const->MatrixKV<double> override{ return term_score_; }
	auto getTermScore(TopicId t_id) const->VectorV<double> override{ return term_score_[t_id]; }

	// 指定トピックの上位return_word_num個の、語彙とスコアを返す
	// [topic][ranking]<vocab, score>
	auto getWordOfTopic(Distribution target, uint return_word_num) const->VectorK< std::vector< std::tuple<std::wstring, double>>> override{ return LDA::getWordOfTopic(target, return_word_num); }
	// [ranking]<vocab, score>
	auto getWordOfTopic(Distribution target, uint return_word_num, TopicId k_id) const->std::vector< std::tuple<std::wstring, double>> override;

	// 指定ドキュメントの上位return_word_num個の、語彙とスコアを返す
	// [doc][ranking]<vocab, score>
	auto getWordOfDocument(uint return_word_num) const->VectorD< std::vector< std::tuple<std::wstring, double>>> override{ return LDA::getWordOfDocument(return_word_num); }
	//[ranking]<vocab, score>
	auto getWordOfDocument(uint return_word_num, DocumentId d_id) const->std::vector< std::tuple<std::wstring, double>> override;

	uint getDocumentNum() const override{ return D_; }
	uint getTopicNum() const override{ return K_; }
	uint getWordNum() const override{ return V_; }

	// get hyper-parameter of topic distribution
	auto getAlpha() const->VectorK<double> override{ return alpha_; }
	// get hyper-parameter of word distribution
	auto getBeta() const->VectorV<double> override{ return beta_; }

	// 
	double getLogLikelihood() const override;

	double getPerplexity() const override{ return std::exp(-getLogLikelihood() / tokens_.size()); }
};

}
#endif
