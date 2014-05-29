#include "lib/LDA/lda.h"
#include "SigUtil/lib/file.hpp"

const int TopicNum = 20;
const int IterationNum = 5;

static const std::wregex url_reg(L"http(s)?://([\\w-]+\\.)+[\\w-]+(/[\\w- ./?%&=]*)?");
static const std::wregex htag_reg(L"#(\\w)+");
static const std::wregex res_reg(L"@(\\w)+");
static const std::wregex noise_reg(L"^[sWvwÖE¥AB*:F;G[|cL`ßo¡.,_| ~~\\-\\^\"h'fúW@!I?H#Ë() () ¢v{}\\[\\]\/ @]+$");
static const std::wregex a_hira_kata_reg(L"^[-ñ@-0-9O-X]$");

std::vector<std::vector<double>> Experiment(std::wstring src_folder, std::wstring out_folder, bool make_new)
{
	using namespace std;
	using sig::uint;

#if USE_SIGNLP
	// eLXg©çf[^Zbgðì¬·éÛÉgp·étB^
	sigtm::FilterSetting filter(true);

	// gpiÌÝè
	filter.addWordClass(signlp::WordClass::¼);
	filter.addWordClass(signlp::WordClass::`e);
	filter.addWordClass(signlp::WordClass::®);
	
	// `ÔfðÍOÌtB^
	filter.setPreFilter([](wstring& str){
		static auto& replace = sig::ZenHanReplace::GetInstance();

		str = regex_replace(str, url_reg, wstring(L""));
		str = regex_replace(str, htag_reg, wstring(L""));
		str = regex_replace(str, res_reg, wstring(L""));
	});

	// `ÔfðÍãÉtB^
	filter.setAftFilter([](wstring& str){
		str = regex_replace(str, noise_reg, wstring(L""));
		str = regex_replace(str, a_hira_kata_reg, wstring(L""));
	});
#endif

	// üÍf[^ì¬ 
	sigtm::InputDataPtr inputdata;

	if(make_new){
		// Vµ­f[^Zbgðì¬(»ÝT|[gµÄ¢éÌÍeLXg©çÌ¶¬)
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
		// ßÉì¬µ½f[^Zbgðgp or ©ªÅwè`®Ìf[^ZbgðpÓ·éê
		inputdata = sigtm::InputData::makeInstance(out_folder);
	}

	cout << "model calculate" << endl;

	auto lda = sigtm::LDA::makeInstance(TopicNum, inputdata);
	uint doc_num = lda->getDocumentNum();

	// wKJn
	lda->update(IterationNum);

	lda->save(sigtm::LDA::Distribution::DOCUMENT, out_folder);
	lda->save(sigtm::LDA::Distribution::TOPIC, out_folder);
	lda->save(sigtm::LDA::Distribution::TERM_SCORE, out_folder);

	// LDAãÌl¨ÔÞxªè
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
	[Tvf[^]
	 2013N8ã¼Ìl¨¼ðNGÆµÄûWµ½tweet (ÎÛl¨Ílist.txtÉLÚ)
	
	E¶(l¨)F77
	Ej[NPêF80296
	EPêF2048595
	*/
	
	std::wstring fpass = L"../SigTM/test data";
	
	auto simirality = Experiment(fpass + L"/src_documents", fpass, true);

	return 0;
}