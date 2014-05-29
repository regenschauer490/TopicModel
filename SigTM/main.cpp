#include "lib/LDA/lda.h"
#include "SigUtil/lib/file.hpp"

const int DocumentNum = 77;
const int TopicNum = 20;
const int IterationNum = 500;

static const std::wregex url_reg(L"http(s)?://([\\w-]+\\.)+[\\w-]+(/[\\w- ./?%&=]*)?");
static const std::wregex htag_reg(L"#(\\w)+");
static const std::wregex res_reg(L"@(\\w)+");
static const std::wregex noise_reg(L"^[sWvwÖE¥AB*:F;G[|cL`ßo¡.,_| ~~\\-\\^\"h'fúW@!I?H#Ë() () ¢v{}\\[\\]\/ @]+$");
static const std::wregex a_hira_kata_reg(L"^[-ñ@-0-9O-X]$");

std::vector<std::vector<double>> Experiment(std::wstring src_folder, std::wstring out_folder, bool make_new)
{
	using sig::uint;
	const uint doc_num = DocumentNum; 

	/*
	sigtm::FilterSetting filter(true);

	//gpiÌÝè
	filter.AddWordClass(signlp::WordClass::¼);
	filter.AddWordClass(WordClass::`e);
	filter.AddWordClass(WordClass::®);
	
	//`ÔfðÍOÌtB^
	filter.SetPreFilter([](wstring& str){
		static auto& replace = sig::ZenHanReplace::GetInstance();

		str = regex_replace(str, url_reg, wstring(L""));
		str = regex_replace(str, htag_reg, wstring(L""));
		str = regex_replace(str, res_reg, wstring(L""));
	});

	//`ÔfðÍãÉtB^
	filter.SetAftFilter([](wstring& str){
		str = regex_replace(str, noise_reg, wstring(L""));
		str = regex_replace(str, a_hira_kata_reg, wstring(L""));
	});
	*/

	// üÍf[^ì¬ 
	sigtm::InputDataPtr inputdata;
	std::vector< std::vector<double> > similarity(doc_num, std::vector<double>(doc_num, 0));

	if(make_new){
		auto doc_pass = sig::get_file_names(src_folder, false);
#if USE_SIGNLP
		vector<vector<wstring>> docs;
		for (auto dp : *doc_pass){
			docs.push_back(sig::STRtoWSTR(*sig::ReadLine<std::string>(sig::DirpassTailModify(src_folder, true) + dp)));
			std::wcout << docs[0][0];
		}
		inputdata = sigtm::InputData::MakeInstance(docs, filter, out_folder);
#endif
	}
	else{
		inputdata = sigtm::InputData::makeInstance(out_folder);
	}

	//wKJn
	std::cout << "model calculate" << std::endl;

	auto lda = sigtm::LDA::makeInstance(TopicNum, inputdata);

	lda->update(IterationNum);

	lda->save(sigtm::LDA::Distribution::DOCUMENT, out_folder);
	lda->save(sigtm::LDA::Distribution::TOPIC, out_folder);
	lda->save(sigtm::LDA::Distribution::TERM_SCORE, out_folder);

	// LDAãÌl¨ÔÞxªè
	for (uint i = 0; i < doc_num; ++i){
		for (uint j = 0; j < i; ++j)	similarity[i][j] = similarity[j][i];
		for (uint j = i; j < doc_num; ++j)	similarity[i][j] = lda->compareDistribution(sigtm::CompareMethodD::JS_DIV, sigtm::LDA::Distribution::DOCUMENT, i, j);
	}

	//sig::SaveCSV(similarity, names, names, out_folder + L"similarity_lda.csv");

	return std::move(similarity);
}


int main()
{
	/*
	[Tvf[^]
	 2013N8ã¼Ìl¨¼ðNGÆµÄûWµ½tweet (ÎÛl¨Ílist.txtÉLÚ)
	
	E¶(l¨)F77
	Ej[NPêF80296
	EPêF2048595
	*/
	
	std::wstring fpass = L"test data";
	
	auto simirality = Experiment(fpass + L"/src_documents", fpass, false);

	return 0;
}