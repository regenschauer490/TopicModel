/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_LDA_HPP
#define SIG_LDA_HPP

#include "../sigtm.hpp"
#include "../helper/input.h"
#include "../helper/compare_method.hpp"

#if USE_SIGNLP
#include "lib/helper/input_text.h"
#endif

#include "SigUtil/lib/tool.hpp"

namespace sigtm
{
class LDA;
typedef std::shared_ptr<LDA> LDAPtr;


/* Latent Dirichlet Allocation (estimate by Collapsed Gibbs Sampling) */
class LDA 
{
public:
	// LDAで得られる確率分布やベクトル
	enum class Distribution{ DOCUMENT, TOPIC, TERM_SCORE };

	SIG_MakeCompareInnerClass(LDA);

private:
	// method chain 生成
	SIG_MakeDist2CmpMapBase;
	SIG_MakeDist2CmpMap(Distribution::DOCUMENT, LDA::CmpD<std::function< std::vector<double>(uint) >>);
	SIG_MakeDist2CmpMap(Distribution::TOPIC, LDA::CmpD<std::function< std::vector<double>(uint) >>);
	SIG_MakeDist2CmpMap(Distribution::TERM_SCORE, LDA::CmpV<std::function< std::vector<double>(uint) >>);
	//SIG_MakeDist2CmpMap(Distribution::DOC_TERM, LDA::CmpV);

private:
	const uint D_NUM;		// number of documents
	const uint T_NUM;		// number of topics
	const uint W_NUM;		// number of words

	// hyper parameter of dirichlet distribution
	const double alpha_;
	const double beta_;
		
	// original input data
	InputDataPtr input_data_;
	std::vector<TokenPtr> const& tokens_;
	std::vector<C_WStrPtr> const& words_;
		
	// implementation variables
	std::vector< std::vector<uint> > word_ct_;	//[W_NUM][T_NUM]
	std::vector< std::vector<uint> > doc_ct_;	//[D_NUM][T_NUM]
	std::vector<uint> topic_ct_;				//[T_NUM]

	std::vector<double> p_;
	std::vector<uint> z_;		//各トークンに(暫定的に)割り当てられたトピック

	std::vector< std::vector<double> > tscore_;	//[T_NUM][W_NUM]
	uint iter_ct_;

	// random generator
	sig::SimpleRandom<uint> rand_ui_;
	sig::SimpleRandom<double> rand_d_;

private:
	LDA() = delete;
	LDA(LDA const&) = delete;

	// alpha_:各単語のトピック更新時の選択確率平滑化定数, beta_:
	LDA(uint topic_num, InputDataPtr input_data, maybe<double> alpha, maybe<double> beta) : 
		D_NUM(input_data->doc_num_), T_NUM(topic_num), W_NUM(input_data->words_.size()), alpha_(alpha ? *alpha : 50.0/T_NUM), beta_(beta ? *beta : 0.1), input_data_(input_data),
		tokens_(input_data->tokens_), words_(input_data->words_), word_ct_(W_NUM, std::vector<uint>(T_NUM, 0)), doc_ct_(D_NUM, std::vector<uint>(T_NUM, 0)), topic_ct_(T_NUM, 0),
		p_(T_NUM, 0.0), z_(tokens_.size(), 0), tscore_(T_NUM, std::vector<double>(W_NUM, 0)), rand_ui_(0, T_NUM - 1, FIXED_RANDOM), rand_d_(0.0, 1.0, FIXED_RANDOM), iter_ct_(0)
	{
		initSetting();
	}
/*		LDA(uint const doc_num, uint const topic_num, uint const word_num, std::vector<Token>&& tok_lis, std::vector<StrPtr>&& word_vec) :
		D_NUM(doc_num), T_NUM(topic_num), W_NUM(word_num), alpha_(50.0/T_NUM), beta_(0.1), tokens_( std::move(tok_lis) ), words_( std::move(word_vec) ),
		word_ct_(W_NUM, std::vector<uint>(T_NUM, 0)), doc_ct_(D_NUM, std::vector<uint>(T_NUM, 0)), topic_ct_(T_NUM, 0),
		p_(T_NUM, 0.0), z_(tokens_.size(), 0),tscore_(T_NUM, std::vector<double>(W_NUM, 0) ),rand_ui_(0, T_NUM-1), rand_d_(0.0, 1.0), iter_ct_(0)
	{
		initSetting();
	}
*/
	void initSetting();
	uint selectNextTopic(TokenPtr const& t);
	void resample(TokenPtr const& t);

	void printTopicWord(Distribution dist, std::wstring const& save_pass) const;
	void printDocumentTopic(std::wstring const& save_pass) const;
	void printDocumentWord(std::wstring const& save_pass) const;

	std::vector< std::tuple<std::wstring, double> > getTopWords(std::vector<double> const& dist, uint num) const;
	void calcTermScore();

public:
	// InputDataで作成した入力データでコンストラクト
	static LDAPtr makeInstance(uint topic_num, InputDataPtr input_data, maybe<double> alpha = nothing, maybe<double> beta = nothing){
		return LDAPtr(new LDA(topic_num, input_data, alpha, beta)); 
	}

/*	// 自前でトークンと語彙リストを作成する場合
	static LDAPtr makeInstance(uint const doc_num, uint const topic_num, uint const word_num, std::vector<Token>&& token_list, std::vector<StrPtr>&& word_list){
		return LDAPtr( new LDA(doc_num, topic_num, word_num, std::move(token_list), std::move(word_list)) ); 
	}*/

	// サンプリングを行い、内部状態を更新する
	void update(uint iteration_num);

	// 確率分布同士の類似度を測る
	// target：トピックorドキュメントの選択, id1,id2：類似度を測る対象のindex, 戻り値：類似度
	double compareDistribution(CompareMethodD method, Distribution target, uint id1, uint id2) const;

	// トピック間の単語分布の類似度を測って似たトピック同士を見つけ、特徴的なトピックのみに圧縮する
	// threshold：閾値, 戻り値：類似トピックの組み合わせ一覧
//	std::vector< std::vector<int> > CompressTopicDimension(CompareMethodD method, double threshold) const;

	void print(Distribution target) const{ save(target, L""); }

	void save(Distribution target, std::wstring const& save_pass) const;

	//ドキュメント毎のトピック選択確率 [doc][topic]
	auto getTheta()->std::vector< std::vector<double> > const;
	auto getTheta(uint document_id)->std::vector<double> const;

	//トピック毎の単語分布 [topic][word]
	auto getPhi()->std::vector< std::vector<double> > const;
	auto getPhi(uint topic_id)->std::vector<double> const;

	//トピックを強調する語スコア [topic][word]
	auto getTermScoreOfTopic()->std::vector< std::vector<double> > const{ return tscore_; }
	auto getTermScoreOfTopic(int t_id)->std::vector<double> const{ return tscore_[t_id]; }

	//ドキュメントのThetaとTermScoreの積 [ranking]<word_id,score>
	auto getTermScoreOfDocument(uint d_id)->std::vector< std::tuple<uint, double> > const;

	// 指定トピックの上位return_word_num個の、語彙とスコアを返す
	// [topic][word]<vocab, score>
	auto getWordOfTopic(Distribution target, uint return_word_num)->std::vector< std::vector< std::tuple<std::wstring, double> > > const;
	// [word]<vocab, score>
	auto getWordOfTopic(Distribution target, uint return_word_num, uint topic_id)->std::vector< std::tuple<std::wstring, double> > const;

	// 指定ドキュメントの上位return_word_num個の、語彙とスコアを返す
	// [doc][word]<vocab, score>
	auto getWordOfDocument(uint return_word_num)->std::vector< std::vector< std::tuple<std::wstring, double> > > const;
	//[doc]<vocab, score>
	auto getWordOfDocument(uint return_word_num, uint doc_id)->std::vector< std::tuple<std::wstring, double> > const;
};

}
#endif
