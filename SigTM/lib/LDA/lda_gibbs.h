/*
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

/* Latent Dirichlet Allocation (estimate by Collapsed Gibbs Sampling) */
class LDA_Gibbs : public LDA
{
	const uint D_;		// number of documents
	const uint K_;		// number of topics
	const uint V_;		// number of words

	// hyper parameter of dirichlet distribution
	const double alpha_;
	const double beta_;
		
	// original input data
	InputDataPtr input_data_;
	TokenList const& tokens_;
		
	// implementation variables
	MatrixVK<uint> word_ct_;	//[V_][K_]
	MatrixDK<uint> doc_ct_;		//[D_][K_]
	VectorK<uint> topic_ct_;	//[K_]

	std::vector<double> p_;
	std::vector<uint> z_;		// 各トークンに(暫定的に)割り当てられたトピック

	MatrixKV<double> term_score_;	//[K_][V_]
	uint iter_ct_;

	// random generator
	sig::SimpleRandom<uint> rand_ui_;
	sig::SimpleRandom<double> rand_d_;

private:
	LDA_Gibbs() = delete;
	LDA_Gibbs(LDA_Gibbs const&) = delete;

	// alpha_:各単語のトピック更新時の選択確率平滑化定数, beta_:
	LDA_Gibbs(uint topic_num, InputDataPtr input_data, maybe<double> alpha, maybe<double> beta) :
		D_(input_data->getDocNum()), K_(topic_num), V_(input_data->getWordNum()), alpha_(alpha ? *alpha : 50.0/K_), beta_(beta ? *beta : 0.1), input_data_(input_data),
		tokens_(input_data->tokens_), word_ct_(V_, VectorK<uint>(K_, 0)), doc_ct_(D_, VectorK<uint>(K_, 0)), topic_ct_(K_, 0),
		p_(K_, 0.0), z_(tokens_.size(), 0), term_score_(K_, VectorV<double>(V_, 0)), rand_ui_(0, K_ - 1, FixedRandom), rand_d_(0.0, 1.0, FixedRandom), iter_ct_(0)
	{
		initSetting();
	}

	void initSetting();
	auto selectNextTopic(Token const& t)->TopicId;
	void resample(Token const& t);

public:
	~LDA_Gibbs(){}

	DynamicType getDynamicType() const override{ return DynamicType::GIBBS; }

	// InputDataで作成した入力データを元にコンストラクト
	static LDAPtr makeInstance(uint topic_num, InputDataPtr input_data, maybe<double> alpha = nothing, maybe<double> beta = nothing){
		return LDAPtr(new LDA_Gibbs(topic_num, input_data, alpha, beta)); 
	}
	
	// モデルの学習を行う
	// iteration_num: 学習の反復回数(MCMCによる全変数の更新を1反復とする)
	void learn(uint iteration_num) override;

	// 確率分布同士の類似度を測る(メソッドチェーンな感じに使用)
	// Select: LDA::Distributionから選択, id1,id2：類似度を測る対象のindex
	// return -> 比較関数の選択(関数オブジェクト)
	template <Distribution Select>
	auto compare(Id id1, Id id2) const->typename Map2Cmp<Select>::type
	{
		return Select == Distribution::DOCUMENT
			? typename Map2Cmp<Select>::type(id1, id2, [this](DocumentId id){ return this->getTopicDistribution(id); }, id1 < D_ && id2 < D_ ? true : false)
			: Select == Distribution::TOPIC
				? typename Map2Cmp<Select>::type(id1, id2, [this](TopicId id){ return this->getWordDistribution(id); }, id1 < K_ && id2 < K_ ? true : false)
				: Select == Distribution::TERM_SCORE
					? typename Map2Cmp<Select>::type(id1, id2, [this](TopicId id){ return this->getTermScoreOfTopic(id); }, id1 < K_ && id2 < K_ ? true : false)
					: typename Map2Cmp<Select>::type(id1, id2, [](TopicId id){ return std::vector<double>(); }, false);
	}

	// トピック間の単語分布の類似度を測って似たトピック同士を見つけ、特徴的なトピックのみに圧縮する
	// threshold：閾値, 戻り値：類似トピックの組み合わせ一覧
//	std::vector< std::vector<int> > CompressTopicDimension(CompareMethodD method, double threshold) const;

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
	auto getTermScoreOfTopic() const->MatrixKV<double> override{ return term_score_; }
	auto getTermScoreOfTopic(TopicId t_id) const->VectorV<double> override{ return term_score_[t_id]; }

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
	auto getAlpha() const->VectorK<double> override{ return sig::replicate(K_, alpha_); }
	// get hyper-parameter of word distribution
	auto getEta() const->VectorV<double> override{ return sig::replicate(V_, beta_); }
};

}
#endif
