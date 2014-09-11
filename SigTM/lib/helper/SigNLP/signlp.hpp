#ifndef SIG_NLP_HPP
#define SIG_NLP_HPP

#include "SigUtil/lib/string.hpp"
#include "SigUtil/lib/file.hpp"

namespace signlp{

#define USE_SIGNLP 1				// 文字列解析を行うためにSigNLPを使用するか

const bool enable_warning = false;

using sig::uint;

enum class WordClass{ NA, 名詞, 動詞, 形容詞, 副詞, 接続詞, 感動詞, 助詞, 助動詞, 連体詞, 記号 };

//Positive, Negative, nEutral
enum class PosiNega { NA, P, N, E };

//【行為】,【評価・感情/主観】,【出来事】,【存在・性質】,【経験】,【場所】,【状態/客観】
enum class PNStandard { NA, Act, EvaEmo_Sbj, Event, ExisProp, Exp, State_Obj, Place };


inline WordClass StrToWC(std::string const& str){
	if (str == "名詞") return WordClass::名詞;
	if (str == "動詞") return WordClass::動詞;
	if (str == "形容詞") return WordClass::形容詞;
	if (str == "副詞") return WordClass::副詞;
	if (str == "感動詞") return WordClass::感動詞;
	if (str == "接続詞") return WordClass::接続詞;
	if (str == "助詞") return WordClass::助詞;
	if (str == "助動詞") return WordClass::助動詞;
	if (str == "連体詞") return WordClass::連体詞;
	if (str == "記号") return WordClass::記号;
	return WordClass::NA;
};

}
#endif