#ifndef SIG_PROCESS_WORD_H
#define SIG_PROCESS_WORD_H

#include <unordered_map>
#include "mecab_wrapper.h"
#include "signlp.hpp"

namespace signlp{

static const wchar_t* evaluation_word_filepass = L"J_Evaluation_lib/�P�ꊴ��ɐ��Ή��\.csv"; //evword.csv";
static const auto evaluation_noun_filepass = std::wstring(L"J_Evaluation_lib/noun.csv");
static const auto evaluation_declinable_filepass = std::wstring(L"J_Evaluation_lib/declinable.csv");
static const double evaluation_word_threshold = 0.80;		//���{��]���ɐ������Ŏg�p�����b��臒l�ݒ�(|�X�R�A|>臒l)


typedef std::unordered_map<WordClass, unsigned> ScoreMap;

/* ���{��]���ɐ������������V���O���g���N���X */
class EvaluationLibrary{
	//�P�ꊴ��ɐ��Ή��\ [-1�`1�̃X�R�A, �i��]
	std::unordered_map< std::string, std::tuple<double, WordClass> > word_ev_;
	//����,�p����P/N���� [P/N, �]���]
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

	/* �P�ꊴ��ɐ��Ή��\�g�p */
	//���͂�P/N�𔻒� (�e�P��-1�`1�̃X�R�A���t�^����Ă���,���v�����߂�. N: total < -threshold�CP: threshold < total) 
	PosiNega getSentencePN(std::string const& sentence, uint threshold = 0.8) const;
	PosiNega getSentencePN(std::wstring const& sentence, uint threshold = 0.8) const;


	/* ����,�p����P/N����g�p */
	//�P���P/N���擾
	PosiNega getWordPN(std::string const&  word, WordClass wc) const;
	PosiNega getWordPN(std::wstring const& word, WordClass wc) const;
	
	//���͂�P/N�𔻒� (�i���ʂ̃X�R�A��臒l��C�ӂɗ^����) 
	PosiNega getSentencePN(std::string const&  sentence, ScoreMap score_map, double th) const;
	PosiNega getSentencePN(std::wstring const& sentence, ScoreMap score_map, double th) const;
		
	//���̒P���P/N�̕]������擾
	PNStandard getPNStandard(std::string const&  word) const;
	PNStandard getPNStandard(std::wstring const& word) const;
};

}
#endif