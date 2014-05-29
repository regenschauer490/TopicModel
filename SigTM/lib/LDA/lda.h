/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_LDA_HPP
#define SIG_LDA_HPP

#include "../sigtm.hpp"
#include "../helper/input.h"
#include "../helper/compare_method.hpp"

#if USE_SIGNLP
#include "../helper/input_text.h"
#endif

#include "SigUtil/lib/tool.hpp"

namespace sigtm
{
class LDA;
typedef std::shared_ptr<LDA> LDAPtr;


/* Latent Dirichlet Allocation (estimate by Collapsed Gibbs Sampling) */
class LDA 
{
public:
	// LDA�œ�����m�����z��x�N�g��
	enum class Distribution{ DOCUMENT, TOPIC, TERM_SCORE };

	SIG_MakeCompareInnerClass(LDA);

private:
	// method chain ����
	SIG_MakeDist2CmpMapBase;
	SIG_MakeDist2CmpMap(Distribution::DOCUMENT, LDA::CmpD<std::function< std::vector<double>(uint) >>);
	SIG_MakeDist2CmpMap(Distribution::TOPIC, LDA::CmpD<std::function< std::vector<double>(uint) >>);
	SIG_MakeDist2CmpMap(Distribution::TERM_SCORE, LDA::CmpV<std::function< std::vector<double>(uint) >>);
	//SIG_MakeDist2CmpMap(Distribution::DOC_TERM, LDA::CmpV);

private:
	const uint D_NUM;		// number of documents
	const uint T_NUM;		// number of topics
	const uint W_NUM;		// number of words

	// hyper parameter of dirichlet distribution
	const double alpha_;
	const double beta_;
		
	// original input data
	InputDataPtr input_data_;
	std::vector<TokenPtr> const& tokens_;
	std::vector<C_WStrPtr> const& words_;
		
	// implementation variables
	std::vector< std::vector<uint> > word_ct_;	//[W_NUM][T_NUM]
	std::vector< std::vector<uint> > doc_ct_;	//[D_NUM][T_NUM]
	std::vector<uint> topic_ct_;				//[T_NUM]

	std::vector<double> p_;
	std::vector<uint> z_;		//�e�g�[�N����(�b��I��)���蓖�Ă�ꂽ�g�s�b�N

	std::vector< std::vector<double> > tscore_;	//[T_NUM][W_NUM]
	uint iter_ct_;

	// random generator
	sig::SimpleRandom<uint> rand_ui_;
	sig::SimpleRandom<double> rand_d_;

private:
	LDA() = delete;
	LDA(LDA const&) = delete;

	// alpha_:�e�P��̃g�s�b�N�X�V���̑I���m���������萔, beta_:
	LDA(uint topic_num, InputDataPtr input_data, maybe<double> alpha, maybe<double> beta) : 
		D_NUM(input_data->doc_num_), T_NUM(topic_num), W_NUM(input_data->words_.size()), alpha_(alpha ? *alpha : 50.0/T_NUM), beta_(beta ? *beta : 0.1), input_data_(input_data),
		tokens_(input_data->tokens_), words_(input_data->words_), word_ct_(W_NUM, std::vector<uint>(T_NUM, 0)), doc_ct_(D_NUM, std::vector<uint>(T_NUM, 0)), topic_ct_(T_NUM, 0),
		p_(T_NUM, 0.0), z_(tokens_.size(), 0), tscore_(T_NUM, std::vector<double>(W_NUM, 0)), rand_ui_(0, T_NUM - 1, FIXED_RANDOM), rand_d_(0.0, 1.0, FIXED_RANDOM), iter_ct_(0)
	{
		initSetting();
	}
/*		LDA(uint const doc_num, uint const topic_num, uint const word_num, std::vector<Token>&& tok_lis, std::vector<StrPtr>&& word_vec) :
		D_NUM(doc_num), T_NUM(topic_num), W_NUM(word_num), alpha_(50.0/T_NUM), beta_(0.1), tokens_( std::move(tok_lis) ), words_( std::move(word_vec) ),
		word_ct_(W_NUM, std::vector<uint>(T_NUM, 0)), doc_ct_(D_NUM, std::vector<uint>(T_NUM, 0)), topic_ct_(T_NUM, 0),
		p_(T_NUM, 0.0), z_(tokens_.size(), 0),tscore_(T_NUM, std::vector<double>(W_NUM, 0) ),rand_ui_(0, T_NUM-1), rand_d_(0.0, 1.0), iter_ct_(0)
	{
		initSetting();
	}
*/
	void initSetting();
	uint selectNextTopic(TokenPtr const& t);
	void resample(TokenPtr const& t);

	void printTopicWord(Distribution dist, std::wstring const& save_pass) const;
	void printDocumentTopic(std::wstring const& save_pass) const;
	void printDocumentWord(std::wstring const& save_pass) const;

	std::vector< std::tuple<std::wstring, double> > getTopWords(std::vector<double> const& dist, uint num) const;
	void calcTermScore();

public:
	// InputData�ō쐬�������̓f�[�^�ŃR���X�g���N�g
	static LDAPtr makeInstance(uint topic_num, InputDataPtr input_data, maybe<double> alpha = nothing, maybe<double> beta = nothing){
		return LDAPtr(new LDA(topic_num, input_data, alpha, beta)); 
	}

	// �T���v�����O���s���A������Ԃ��X�V����
	void update(uint iteration_num);

	// �m�����z���m�̗ގ��x�𑪂�
	// target�F�g�s�b�Nor�h�L�������g�̑I��, id1,id2�F�ގ��x�𑪂�Ώۂ�index, �߂�l�F�ގ��x
//	double compareDistribution(CompareMethodD method, Distribution target, uint id1, uint id2) const;

	template <Distribution Select>
	auto compare(uint id1, uint id2) const->typename Map2Cmp<Select>::type{
		return Select == Distribution::DOCUMENT
			? typename Map2Cmp<Select>::type(id1, id2, [this](uint id){ return this->getTheta(id); }, id1 < D_NUM && id2 < D_NUM ? true : false)
			: Select == Distribution::TOPIC
				? typename Map2Cmp<Select>::type(id1, id2, [this](uint id){ return this->getPhi(id); }, id1 < T_NUM && id2 < T_NUM ? true : false)
				: Select == Distribution::TERM_SCORE
					? typename Map2Cmp<Select>::type(id1, id2, [this](uint id){ return this->getTermScoreOfTopic(id); }, id1 < T_NUM && id2 < T_NUM ? true : false)
					: typename Map2Cmp<Select>::type(id1, id2, [](uint id){ return std::vector<double>(); }, false);
	}

	// �g�s�b�N�Ԃ̒P�ꕪ�z�̗ގ��x�𑪂��Ď����g�s�b�N���m�������A�����I�ȃg�s�b�N�݂̂Ɉ��k����
	// threshold�F臒l, �߂�l�F�ގ��g�s�b�N�̑g�ݍ��킹�ꗗ
//	std::vector< std::vector<int> > CompressTopicDimension(CompareMethodD method, double threshold) const;

	void print(Distribution target) const{ save(target, L""); }

	void save(Distribution target, std::wstring const& save_pass) const;

	//�h�L�������g���̃g�s�b�N�I���m�� [doc][topic]
	auto getTheta() const->std::vector< std::vector<double> >;
	auto getTheta(uint document_id) const->std::vector<double>;

	//�g�s�b�N���̒P�ꕪ�z [topic][word]
	auto getPhi() const->std::vector< std::vector<double> >;
	auto getPhi(uint topic_id) const->std::vector<double>;

	//�g�s�b�N�����������X�R�A [topic][word]
	auto getTermScoreOfTopic() const->std::vector< std::vector<double> >{ return tscore_; }
	auto getTermScoreOfTopic(int t_id) const->std::vector<double>{ return tscore_[t_id]; }

	//�h�L�������g��Theta��TermScore�̐� [ranking]<word_id,score>
	auto getTermScoreOfDocument(uint d_id) const->std::vector< std::tuple<uint, double> >;

	// �w��g�s�b�N�̏��return_word_num�́A��b�ƃX�R�A��Ԃ�
	// [topic][word]<vocab, score>
	auto getWordOfTopic(Distribution target, uint return_word_num) const->std::vector< std::vector< std::tuple<std::wstring, double> > >;
	// [word]<vocab, score>
	auto getWordOfTopic(Distribution target, uint return_word_num, uint topic_id) const->std::vector< std::tuple<std::wstring, double> >;

	// �w��h�L�������g�̏��return_word_num�́A��b�ƃX�R�A��Ԃ�
	// [doc][word]<vocab, score>
	auto getWordOfDocument(uint return_word_num) const->std::vector< std::vector< std::tuple<std::wstring, double> > >;
	//[doc]<vocab, score>
	auto getWordOfDocument(uint return_word_num, uint doc_id) const->std::vector< std::tuple<std::wstring, double> >;

	uint getDocumentNum() const{ return D_NUM; }
	uint getTopicNum() const{ return T_NUM; }
	uint getWordNum() const{ return W_NUM; }
};

}
#endif
