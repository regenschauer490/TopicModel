#ifndef SIG_NLP_HPP
#define SIG_NLP_HPP

#include "SigUtil/lib/string.hpp"
#include "SigUtil/lib/file.hpp"

namespace signlp{
	
const bool enable_warning = false;

using sig::uint;

enum class WordClass{ NA, –¼Œ, “®Œ, Œ`—eŒ, •›Œ, Ú‘±Œ, Š´“®Œ, •Œ, •“®Œ, ˜A‘ÌŒ, ‹L† };

//Positive, Negative, nEutral
enum class PosiNega { NA, P, N, E };

//ysˆ×z,y•]‰¿EŠ´î/åŠÏz,yo—ˆ–z,y‘¶İE«¿z,yŒoŒ±z,yêŠz,yó‘Ô/‹qŠÏz
enum class PNStandard { NA, Act, EvaEmo_Sbj, Event, ExisProp, Exp, State_Obj, Place };


inline WordClass StrToWC(std::string const& str){
	if (str == "–¼Œ") return WordClass::–¼Œ;
	if (str == "“®Œ") return WordClass::“®Œ;
	if (str == "Œ`—eŒ") return WordClass::Œ`—eŒ;
	if (str == "•›Œ") return WordClass::•›Œ;
	if (str == "Š´“®Œ") return WordClass::Š´“®Œ;
	if (str == "Ú‘±Œ") return WordClass::Ú‘±Œ;
	if (str == "•Œ") return WordClass::•Œ;
	if (str == "•“®Œ") return WordClass::•“®Œ;
	if (str == "˜A‘ÌŒ") return WordClass::˜A‘ÌŒ;
	if (str == "‹L†") return WordClass::‹L†;
	return WordClass::NA;
};

}
#endif