#ifndef SIG_NLP_HPP
#define SIG_NLP_HPP

#include "SigUtil/lib/string.hpp"
#include "SigUtil/lib/file.hpp"

namespace signlp
{

#define SIG_USE_MECAB 1

const bool enable_warning = false;

using sig::uint;
using sig::FilepassString;

enum class WordClass{ NA, Noun, Verb, Adjective, Adverb, Conjunction, Interjection, PostParticle, AuxiliaryVerb, Determiner, Symbol };

//Positive, Negative, nEutral
enum class PosiNega { NA, P, N, E };

//【行為】,【評価・感情/主観】,【出来事】,【存在・性質】,【経験】,【場所】,【状態/客観】
enum class PNStandard { NA, Act, EvaEmo_Sbj, Event, ExisProp, Exp, State_Obj, Place };


inline WordClass StrToWC(std::string const& str){
	if (str == "名詞") return WordClass::Noun;
	if (str == "動詞") return WordClass::Verb;
	if (str == "形容詞") return WordClass::Adjective;
	if (str == "副詞") return WordClass::Adverb;
	if (str == "接続詞") return WordClass::Conjunction;
	if (str == "感動詞") return WordClass::Interjection;
	if (str == "助詞") return WordClass::PostParticle;
	if (str == "助動詞") return WordClass::AuxiliaryVerb;
	if (str == "連体詞") return WordClass::Determiner;
	if (str == "記号") return WordClass::Symbol;
	return WordClass::NA;
};

}
#endif