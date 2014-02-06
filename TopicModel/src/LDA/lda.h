#pragma once

#include "../helper/input_container.h"
#include "../helper/compare_method.hpp"

namespace sigdm{
	
class LDA;
typedef std::shared_ptr<LDA> LDAPtr;


/* Latent Dirichlet Allocation (estimate by Gibbs Sampling) を行うクラス */
class LDA 
{
public:
	//LDAで得られる確率分布やベクトル
	enum class Distribution{ DOCUMENT, TOPIC, TERM_SCORE };

	MakeCompareInnerClass(LDA);

private:
	MakeDist2CmpMapBase;
	MakeDist2CmpMap(Distribution::DOCUMENT, LDA::CmpD<std::function< std::vector<double>(uint) >>);
	MakeDist2CmpMap(Distribution::TOPIC, LDA::CmpD<std::function< std::vector<double>(uint) >>);
	MakeDist2CmpMap(Distribution::TERM_SCORE, LDA::CmpV<std::function< std::vector<double>(uint) >>);
	//MakeDist2CmpMap(Distribution::DOC_TERM, LDA::CmpV);

private:
	const uint D_NUM;		//ドキュメント数
	const uint T_NUM;		//トピック数
	const uint W_NUM;		//語彙数

	//hyper parameter
	const double _alpha;
	const double _beta;
		
	//original input data
	InputDataPtr _input_data;
	std::vector<TokenPtr> const& _tokens;
	std::vector<C_WStrPtr> const& _words;
		
	//implementation variables
	std::vector< std::vector<uint> > _word_ct;	//[W_NUM][T_NUM]
	std::vector< std::vector<uint> > _doc_ct;	//[D_NUM][T_NUM]
	std::vector<uint> _topic_ct;				//[T_NUM]

	std::vector<double> _p;
	std::vector<uint> _z;		//各トークンに(暫定的に)割り当てられたトピック

	std::vector< std::vector<double> > _tscore;	//[T_NUM][W_NUM]

	//random generator
	sig::SimpleRandom<uint> _rand_ui;
	sig::SimpleRandom<double> _rand_d;

	uint _iter_ct;

private:
	LDA();// = delete;
	LDA(LDA const&);// = delete;
	LDA& operator=(LDA const&);// = delete;

	//_alpha:各単語のトピック更新時の選択確率平滑化定数, _beta:
	LDA(uint const topic_num, InputDataPtr const& input_data) : 
		D_NUM(input_data->_doc_num), T_NUM(topic_num), W_NUM(input_data->_words.size()), _alpha(50.0/T_NUM), _beta(0.1), _input_data(input_data),
		_tokens(input_data->_tokens), _words(input_data->_words), _word_ct(W_NUM, std::vector<uint>(T_NUM, 0)), _doc_ct(D_NUM, std::vector<uint>(T_NUM, 0)), _topic_ct(T_NUM, 0),
		_p(T_NUM, 0.0), _z(_tokens.size(), 0), _tscore(T_NUM, std::vector<double>(W_NUM, 0)), _rand_ui(0, T_NUM - 1, FIXED_RANDOM), _rand_d(0.0, 1.0, FIXED_RANDOM), _iter_ct(0)
	{
		Init_();
	}
/*		LDA(uint const doc_num, uint const topic_num, uint const word_num, std::vector<Token>&& tok_lis, std::vector<StrPtr>&& word_vec) :
		D_NUM(doc_num), T_NUM(topic_num), W_NUM(word_num), _alpha(50.0/T_NUM), _beta(0.1), _tokens( std::move(tok_lis) ), _words( std::move(word_vec) ),
		_word_ct(W_NUM, std::vector<uint>(T_NUM, 0)), _doc_ct(D_NUM, std::vector<uint>(T_NUM, 0)), _topic_ct(T_NUM, 0),
		_p(T_NUM, 0.0), _z(_tokens.size(), 0),_tscore(T_NUM, std::vector<double>(W_NUM, 0) ),_rand_ui(0, T_NUM-1), _rand_d(0.0, 1.0), _iter_ct(0)
	{
		Init_();
	}
*/
	void Init_();
	uint SelectNextTopic_(TokenPtr const& t);
	void Resample_(TokenPtr const& t);

	void PrintTopicWord_(Distribution dist, std::wstring const& save_pass) const;
	void PrintDocumentTopic_(std::wstring const& save_pass) const;
	void PrintDocumentWord_(std::wstring const& save_pass) const;

	std::vector< std::tuple<std::wstring, double> > GetTopWords_(std::vector<double> const& dist, uint num) const;
	void CalcTermScore_();

public:
	//InputDataFactoryで作成した入力データでコンストラクト (推奨) 
	static LDAPtr MakeInstance(uint topic_num, InputDataPtr input_data){ return LDAPtr(new LDA(topic_num, input_data)); }

/*	//自前でトークンと語彙リストを作成する場合
	static LDAPtr MakeInstance(uint const doc_num, uint const topic_num, uint const word_num, std::vector<Token>&& token_list, std::vector<StrPtr>&& word_list){
		return LDAPtr( new LDA(doc_num, topic_num, word_num, std::move(token_list), std::move(word_list)) ); 
	}*/

	~LDA(){}

	//サンプリングを行い、内部状態を更新する
	void Update(uint iteration_num);

	//確率分布同士の類似度を測る
	//target：トピックorドキュメントの選択, id1,id2：類似度を測る対象のindex, 戻り値：類似度
	double CompareDistribution(CompareMethodD method, Distribution target, uint id1, uint id2) const;

	//トピック間の単語分布の類似度を測って似たトピック同士を見つけ、特徴的なトピックのみに圧縮する
	//threshold：閾値, 戻り値：類似トピックの組み合わせ一覧
//	std::vector< std::vector<int> > CompressTopicDimension(CompareMethodD method, double threshold) const;

	void Print(Distribution target) const{ Save(target, L""); }

	void Save(Distribution target, std::wstring const& save_pass) const;

	//ドキュメント毎のトピック選択確率 [doc][topic]
	std::vector< std::vector<double> > GetTheta() const;
	std::vector<double> GetTheta(uint document_id) const;

	//トピック毎の単語分布 [topic][word]
	std::vector< std::vector<double> > GetPhi() const;
	std::vector<double> GetPhi(uint topic_id) const;

	//トピックを強調する語スコア [topic][word]
	std::vector< std::vector<double> > GetTermScore() const{ return _tscore; }
	std::vector<double> GetTermScore(int t_id) const{ return _tscore[t_id]; }

	//ドキュメントのThetaとTermScoreの積 [rank]<word_id,score>
	std::vector< std::tuple<uint, double> > GetDocTermScore(uint d_id) const;

	//指定トピックの上位return_word_num個の、語彙とスコアを返す
	//[topic][word]<vocab, score>
	std::vector< std::vector< std::tuple<std::wstring, double> > > GetTopicWord(Distribution target, uint return_word_num) const;
	//[word]<vocab, score>
	std::vector< std::tuple<std::wstring, double> > GetTopicWord(Distribution target, uint return_word_num, uint topic_id) const;

	//指定ドキュメントの上位return_word_num個の、語彙とスコアを返す
	//[doc][word]<vocab, score>
	std::vector< std::vector< std::tuple<std::wstring, double> > > GetDocumentWord(uint return_word_num) const;
	//[doc]<vocab, score>
	std::vector< std::tuple<std::wstring, double> > GetDocumentWord(uint return_word_num, uint doc_id) const;
};

}	//namespace sigdm
