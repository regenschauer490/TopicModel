#pragma once

#include "../helper/input_container.h"
#include "../helper/compare_method.hpp"

namespace sigdm{
	
class LDA;
typedef std::shared_ptr<LDA> LDAPtr;


/* Latent Dirichlet Allocation (estimate by Gibbs Sampling) ���s���N���X */
class LDA 
{
public:
	//LDA�œ�����m�����z��x�N�g��
	enum class Distribution{ DOCUMENT, TOPIC, TERM_SCORE };

	MakeCompareInnerClass(LDA);

private:
	MakeDist2CmpMapBase;
	MakeDist2CmpMap(Distribution::DOCUMENT, LDA::CmpD<std::function< std::vector<double>(uint) >>);
	MakeDist2CmpMap(Distribution::TOPIC, LDA::CmpD<std::function< std::vector<double>(uint) >>);
	MakeDist2CmpMap(Distribution::TERM_SCORE, LDA::CmpV<std::function< std::vector<double>(uint) >>);
	//MakeDist2CmpMap(Distribution::DOC_TERM, LDA::CmpV);

private:
	const uint D_NUM;		//�h�L�������g��
	const uint T_NUM;		//�g�s�b�N��
	const uint W_NUM;		//��b��

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
	std::vector<uint> _z;		//�e�g�[�N����(�b��I��)���蓖�Ă�ꂽ�g�s�b�N

	std::vector< std::vector<double> > _tscore;	//[T_NUM][W_NUM]

	//random generator
	sig::SimpleRandom<uint> _rand_ui;
	sig::SimpleRandom<double> _rand_d;

	uint _iter_ct;

private:
	LDA();// = delete;
	LDA(LDA const&);// = delete;
	LDA& operator=(LDA const&);// = delete;

	//_alpha:�e�P��̃g�s�b�N�X�V���̑I���m���������萔, _beta:
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
	//InputDataFactory�ō쐬�������̓f�[�^�ŃR���X�g���N�g (����) 
	static LDAPtr MakeInstance(uint topic_num, InputDataPtr input_data){ return LDAPtr(new LDA(topic_num, input_data)); }

/*	//���O�Ńg�[�N���ƌ�b���X�g���쐬����ꍇ
	static LDAPtr MakeInstance(uint const doc_num, uint const topic_num, uint const word_num, std::vector<Token>&& token_list, std::vector<StrPtr>&& word_list){
		return LDAPtr( new LDA(doc_num, topic_num, word_num, std::move(token_list), std::move(word_list)) ); 
	}*/

	~LDA(){}

	//�T���v�����O���s���A������Ԃ��X�V����
	void Update(uint iteration_num);

	//�m�����z���m�̗ގ��x�𑪂�
	//target�F�g�s�b�Nor�h�L�������g�̑I��, id1,id2�F�ގ��x�𑪂�Ώۂ�index, �߂�l�F�ގ��x
	double CompareDistribution(CompareMethodD method, Distribution target, uint id1, uint id2) const;

	//�g�s�b�N�Ԃ̒P�ꕪ�z�̗ގ��x�𑪂��Ď����g�s�b�N���m�������A�����I�ȃg�s�b�N�݂̂Ɉ��k����
	//threshold�F臒l, �߂�l�F�ގ��g�s�b�N�̑g�ݍ��킹�ꗗ
//	std::vector< std::vector<int> > CompressTopicDimension(CompareMethodD method, double threshold) const;

	void Print(Distribution target) const{ Save(target, L""); }

	void Save(Distribution target, std::wstring const& save_pass) const;

	//�h�L�������g���̃g�s�b�N�I���m�� [doc][topic]
	std::vector< std::vector<double> > GetTheta() const;
	std::vector<double> GetTheta(uint document_id) const;

	//�g�s�b�N���̒P�ꕪ�z [topic][word]
	std::vector< std::vector<double> > GetPhi() const;
	std::vector<double> GetPhi(uint topic_id) const;

	//�g�s�b�N�����������X�R�A [topic][word]
	std::vector< std::vector<double> > GetTermScore() const{ return _tscore; }
	std::vector<double> GetTermScore(int t_id) const{ return _tscore[t_id]; }

	//�h�L�������g��Theta��TermScore�̐� [rank]<word_id,score>
	std::vector< std::tuple<uint, double> > GetDocTermScore(uint d_id) const;

	//�w��g�s�b�N�̏��return_word_num�́A��b�ƃX�R�A��Ԃ�
	//[topic][word]<vocab, score>
	std::vector< std::vector< std::tuple<std::wstring, double> > > GetTopicWord(Distribution target, uint return_word_num) const;
	//[word]<vocab, score>
	std::vector< std::tuple<std::wstring, double> > GetTopicWord(Distribution target, uint return_word_num, uint topic_id) const;

	//�w��h�L�������g�̏��return_word_num�́A��b�ƃX�R�A��Ԃ�
	//[doc][word]<vocab, score>
	std::vector< std::vector< std::tuple<std::wstring, double> > > GetDocumentWord(uint return_word_num) const;
	//[doc]<vocab, score>
	std::vector< std::tuple<std::wstring, double> > GetDocumentWord(uint return_word_num, uint doc_id) const;
};

}	//namespace sigdm
