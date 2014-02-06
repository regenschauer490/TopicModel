#ifndef __SIG_NLP_H__
#define __SIG_NLP_H__

#include "helper/SigUtil/lib/sigutil.hpp"
#include "helper/SigUtil/lib/string.hpp"
#include "helper/SigUtil/lib/file.hpp"

namespace signlp{
	
const bool enable_warning = false;

enum class WordClass{ _NA, ����, ����, �`�e��, ����, �ڑ���, ������, ����, ������, �A�̎�, �L�� };

//Positive, Negative, nEutral
enum class PosiNega { _NA, P, N, E };

//�y�s�ׁz,�y�]���E����/��ρz,�y�o�����z,�y���݁E�����z,�y�o���z,�y�ꏊ�z,�y���/�q�ρz
enum class PNStandard { _NA, Act, EvaEmo_Sbj, Event, ExisProp, Exp, State_Obj, Place };


inline WordClass StrToWC(std::string const& str){
	if (str == "����") return WordClass::����;
	if (str == "����") return WordClass::����;
	if (str == "�`�e��") return WordClass::�`�e��;
	if (str == "����") return WordClass::����;
	if (str == "������") return WordClass::������;
	if (str == "�ڑ���") return WordClass::�ڑ���;
	if (str == "����") return WordClass::����;
	if (str == "������") return WordClass::������;
	if (str == "�A�̎�") return WordClass::�A�̎�;
	if (str == "�L��") return WordClass::�L��;
	return WordClass::_NA;
};

}

#endif