#ifndef SIG_PROCESS_WORD_H
#define SIG_PROCESS_WORD_H

#include <unordered_map>
#include "mecab_wrapper.h"
#include "signlp.hpp"

namespace signlp{

static const wchar_t* evaluation_word_filepass = L"J_Evaluation_lib/単語感情極性対応表.csv"; //evword.csv";
static const auto evaluation_noun_filepass = std::wstring(L"J_Evaluation_lib/noun.csv");
static const auto evaluation_declinable_filepass = std::wstring(L"J_Evaluation_lib/declinable.csv");
static const double evaluation_word_threshold = 0.80;		//日本語評価極性辞書で使用する語彙の閾値設定(|スコア|>閾値)


typedef std::unordered_map<WordClass, unsigned> ScoreMap;

/* 日本語評価極性辞書を扱うシングルトンクラス */
class EvaluationLibrary{
	//単語感情極性対応表 [-1～1のスコア, 品詞]
	std::unordered_map< std::string, std::tuple<double, WordClass> > word_ev_;
	//名詞,用言のP/N判定 [P/N, 評価基準]
	std::unordered_map< std::string, std::tuple<PosiNega, PNStandard> > noun_ev_;
	std::unordered_map< std::string, std::tuple<PosiNega, PNStandard> > declinable_ev_;

	MecabWrapper& mecab_;

private:
	EvaluationLibrary();

	EvaluationLibrary(const EvaluationLibrary&) = delete;
public:
	static EvaluationLibrary& getInstance(){
		static EvaluationLibrary instance;
		return instance;
	}

	/* 単語感情極性対応表使用 */
	//文章のP/Nを判定 (各単語-1～1のスコアが付与されており,合計を求める. N: total < -threshold，P: threshold < total) 
	PosiNega getSentencePN(std::string const& sentence, uint threshold = 0.8) const;
	PosiNega getSentencePN(std::wstring const& sentence, uint threshold = 0.8) const;


	/* 名詞,用言のP/N判定使用 */
	//単語のP/Nを取得
	PosiNega getWordPN(std::string const&  word, WordClass wc) const;
	PosiNega getWordPN(std::wstring const& word, WordClass wc) const;
	
	//文章のP/Nを判定 (品詞別のスコアと閾値を任意に与える) 
	PosiNega getSentencePN(std::string const&  sentence, ScoreMap score_map, double th) const;
	PosiNega getSentencePN(std::wstring const& sentence, ScoreMap score_map, double th) const;
		
	//その単語のP/Nの評価基準を取得
	PNStandard getPNStandard(std::string const&  word) const;
	PNStandard getPNStandard(std::wstring const& word) const;
};

}
#endif