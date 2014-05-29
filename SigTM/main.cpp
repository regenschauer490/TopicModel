#include "lib/LDA/lda.h"
#include "SigUtil/lib/file.hpp"

const int TopicNum = 20;
const int IterationNum = 5;

static const std::wregex url_reg(L"http(s)?://([\\w-]+\\.)+[\\w-]+(/[\\w- ./?%&=]*)?");
static const std::wregex htag_reg(L"#(\\w)+");
static const std::wregex res_reg(L"@(\\w)+");
static const std::wregex noise_reg(L"^[�sW�vw���ցE��A�B*��:�F;�G�[�|�c�L`�o�.,_|��~~\\-\\^\"�h'�f�W@!�I?�H#��() () ��v{}\\[\\]\/ �@]+$");
static const std::wregex a_hira_kata_reg(L"^[��-��@-��0-9�O-�X]$");

std::vector<std::vector<double>> Experiment(std::wstring src_folder, std::wstring out_folder, bool make_new)
{
	using namespace std;
	using sig::uint;

#if USE_SIGNLP
	// �e�L�X�g����f�[�^�Z�b�g���쐬����ۂɎg�p����t�B���^
	sigtm::FilterSetting filter(true);

	// �g�p�i���̐ݒ�
	filter.addWordClass(signlp::WordClass::����);
	filter.addWordClass(signlp::WordClass::�`�e��);
	filter.addWordClass(signlp::WordClass::����);
	
	// �`�ԑf��͑O�̃t�B���^����
	filter.setPreFilter([](wstring& str){
		static auto& replace = sig::ZenHanReplace::GetInstance();

		str = regex_replace(str, url_reg, wstring(L""));
		str = regex_replace(str, htag_reg, wstring(L""));
		str = regex_replace(str, res_reg, wstring(L""));
	});

	// �`�ԑf��͌�Ƀt�B���^����
	filter.setAftFilter([](wstring& str){
		str = regex_replace(str, noise_reg, wstring(L""));
		str = regex_replace(str, a_hira_kata_reg, wstring(L""));
	});
#endif

	// ���̓f�[�^�쐬 
	sigtm::InputDataPtr inputdata;

	if(make_new){
		// �V�����f�[�^�Z�b�g���쐬(���݃T�|�[�g���Ă���̂̓e�L�X�g����̐���)
#if USE_SIGNLP
		auto doc_pass = sig::get_file_names(src_folder, false);

		vector<vector<wstring>> docs;
		for (auto dp : sig::fromJust(doc_pass)){
			auto tdoc = sig::read_line<wstring>(sig::impl::modify_dirpass_tail(src_folder, true) + dp);
			docs.push_back(
				sig::fromJust(tdoc)
			);
		}
		inputdata = sigtm::InputDataFromText::makeInstance(docs, filter, out_folder);
#else
		assert(false);
#endif
	}
	else{
		// �ߋ��ɍ쐬�����f�[�^�Z�b�g���g�p or �����Ŏw��`���̃f�[�^�Z�b�g��p�ӂ���ꍇ
		inputdata = sigtm::InputData::makeInstance(out_folder);
	}

	cout << "model calculate" << endl;

	auto lda = sigtm::LDA::makeInstance(TopicNum, inputdata);
	uint doc_num = lda->getDocumentNum();

	// �w�K�J�n
	lda->update(IterationNum);

	lda->save(sigtm::LDA::Distribution::DOCUMENT, out_folder);
	lda->save(sigtm::LDA::Distribution::TOPIC, out_folder);
	lda->save(sigtm::LDA::Distribution::TERM_SCORE, out_folder);

	// LDA��̐l���ԗގ��x����
	vector< vector<double> > similarity(doc_num, vector<double>(doc_num, 0));

	for (uint i = 0; i < doc_num; ++i){
		std::cout << "i:" << i << std::endl;
		for (uint j = 0; j < i; ++j)	similarity[i][j] = similarity[j][i];
		for (uint j = i; j < doc_num; ++j){
			std::cout << " j:" << j << std::endl;
			similarity[i][j] = sig::fromJust(lda->compare<sigtm::LDA::Distribution::DOCUMENT>(i, j).method(sigtm::CompareMethodD::JS_DIV));
		}
		//lda->compareDistribution(sigtm::CompareMethodD::JS_DIV, sigtm::LDA::Distribution::DOCUMENT, i, j);
	}
	
	//sig::SaveCSV(similarity, names, names, out_folder + L"similarity_lda.csv");

	return move(similarity);
}


int main()
{
	/*
	[�T���v���f�[�^]
	 2013�N8���㔼�̐l�������N�G���Ƃ��Ď��W����tweet (�Ώېl����list.txt�ɋL��)
	
	�E������(�l����)�F77
	�E���j�[�N�P�ꐔ�F80296
	�E���P�ꐔ�F2048595
	*/
	
	std::wstring fpass = L"../SigTM/test data";
	
	auto simirality = Experiment(fpass + L"/src_documents", fpass, true);

	return 0;
}