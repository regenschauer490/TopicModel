#ifndef __SENTIMENT_EVALUATION_H__
#define __PROCESS_WORD_H__

#include <unordered_map>

#include "mecab_wrapper.h"
#include "signlp.hpp"

namespace signlp{

static const char* evaluation_word_filepass = "./J_Evaluation_lib/単語感情極性対応表.csv"; //evword.csv";
static const char* evaluation_noun_filepass = "./J_Evaluation_lib/noun.csv";
static const char* evaluation_declinable_filepass = "./J_Evaluation_lib/declinable.csv";
static const double evaluation_word_threshold = 0.80;		//日本語評価極性辞書で使用する語彙の閾値設定(|スコア|>閾値)


typedef std::unordered_map<WordClass, unsigned> ScoreMap;

/* 日本語評価極性辞書を扱うシングルトンクラス */
class EvaluationLibrary{
	//単語感情極性対応表 [-1～1のスコア, 品詞]
	std::unordered_map< std::string, std::tuple<double, WordClass> > _word_ev;
	//名詞,用言のP/N判定 [P/N, 評価基準]
	std::unordered_map< std::string, std::tuple<PosiNega, PNStandard> > _noun_ev;
	std::unordered_map< std::string, std::tuple<PosiNega, PNStandard> > _declinable_ev;

	MecabWrapper& _mecabw;

private:
	EvaluationLibrary();

	EvaluationLibrary(const EvaluationLibrary&) = delete;
public:
	static EvaluationLibrary& GetInstance(){
		static EvaluationLibrary instance;
		return instance;
	}

	/* 単語感情極性対応表使用 */
	//文章のP/Nを判定 (各単語-1～1のスコアが付与されており,合計を求める. 閾値-0.5以下N，0.5以上P) 
	PosiNega GetSentencePosiNega(const std::string& sentence) const;
	PosiNega GetSentencePosiNega(const std::wstring& sentence) const;


	/* 名詞,用言のP/N判定使用 */
	//単語のP/Nを取得
	PosiNega GetWordPosiNega(const std::string& word, WordClass wc) const;
	PosiNega GetWordPosiNega(const std::wstring& word, WordClass wc) const;
	
	//文章のP/Nを判定 (品詞別のスコアと閾値を任意に与える) 
	PosiNega GetSentencePosiNega(const std::string& sentence, ScoreMap score_map, double th) const;
	PosiNega GetSentencePosiNega(const std::wstring& sentence, ScoreMap score_map, double th) const;
		
	//その単語のP/Nの評価基準を取得
	PNStandard GetPNStandard(const std::string& word) const;
	PNStandard GetPNStandard(const std::wstring& word) const;
};

}	//namespace procwoed

#endif