#ifndef __SENTIMENT_EVALUATION_H__
#define __PROCESS_WORD_H__

#include <unordered_map>

#include "mecab_wrapper.h"
#include "signlp.hpp"

namespace signlp{

static const char* evaluation_word_filepass = "./J_Evaluation_lib/�P�ꊴ��ɐ��Ή��\.csv"; //evword.csv";
static const char* evaluation_noun_filepass = "./J_Evaluation_lib/noun.csv";
static const char* evaluation_declinable_filepass = "./J_Evaluation_lib/declinable.csv";
static const double evaluation_word_threshold = 0.80;		//���{��]���ɐ������Ŏg�p�����b��臒l�ݒ�(|�X�R�A|>臒l)


typedef std::unordered_map<WordClass, unsigned> ScoreMap;

/* ���{��]���ɐ������������V���O���g���N���X */
class EvaluationLibrary{
	//�P�ꊴ��ɐ��Ή��\ [-1�`1�̃X�R�A, �i��]
	std::unordered_map< std::string, std::tuple<double, WordClass> > _word_ev;
	//����,�p����P/N���� [P/N, �]���]
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

	/* �P�ꊴ��ɐ��Ή��\�g�p */
	//���͂�P/N�𔻒� (�e�P��-1�`1�̃X�R�A���t�^����Ă���,���v�����߂�. 臒l-0.5�ȉ�N�C0.5�ȏ�P) 
	PosiNega GetSentencePosiNega(const std::string& sentence) const;
	PosiNega GetSentencePosiNega(const std::wstring& sentence) const;


	/* ����,�p����P/N����g�p */
	//�P���P/N���擾
	PosiNega GetWordPosiNega(const std::string& word, WordClass wc) const;
	PosiNega GetWordPosiNega(const std::wstring& word, WordClass wc) const;
	
	//���͂�P/N�𔻒� (�i���ʂ̃X�R�A��臒l��C�ӂɗ^����) 
	PosiNega GetSentencePosiNega(const std::string& sentence, ScoreMap score_map, double th) const;
	PosiNega GetSentencePosiNega(const std::wstring& sentence, ScoreMap score_map, double th) const;
		
	//���̒P���P/N�̕]������擾
	PNStandard GetPNStandard(const std::string& word) const;
	PNStandard GetPNStandard(const std::wstring& word) const;
};

}	//namespace procwoed

#endif