#include "lib/LDA/lda.h"
#include "SigUtil/lib/file.hpp"

const int DocumentNum = 77;
const int TopicNum = 20;
const int IterationNum = 500;

static const std::wregex url_reg(L"http(s)?://([\\w-]+\\.)+[\\w-]+(/[\\w- ./?%&=]*)?");
static const std::wregex htag_reg(L"#(\\w)+");
static const std::wregex res_reg(L"@(\\w)+");
static const std::wregex noise_reg(L"^[ＴWＷwｗω・･、。*＊:：;；ー－…´`ﾟo｡.,_|│~~\\-\\^\"”'’＂@!！?？#⇒() () ｢」{}\\[\\]\/ 　]+$");
static const std::wregex a_hira_kata_reg(L"^[ぁ-んァ-ン0-9０-９]$");

std::vector<std::vector<double>> Experiment(std::wstring src_folder, std::wstring out_folder, bool make_new)
{
	using sig::uint;
	const uint doc_num = DocumentNum; 

	/*
	sigtm::FilterSetting filter(true);

	//使用品詞の設定
	filter.AddWordClass(signlp::WordClass::名詞);
	filter.AddWordClass(WordClass::形容詞);
	filter.AddWordClass(WordClass::動詞);
	
	//形態素解析前のフィルタ処理
	filter.SetPreFilter([](wstring& str){
		static auto& replace = sig::ZenHanReplace::GetInstance();

		str = regex_replace(str, url_reg, wstring(L""));
		str = regex_replace(str, htag_reg, wstring(L""));
		str = regex_replace(str, res_reg, wstring(L""));
	});

	//形態素解析後にフィルタ処理
	filter.SetAftFilter([](wstring& str){
		str = regex_replace(str, noise_reg, wstring(L""));
		str = regex_replace(str, a_hira_kata_reg, wstring(L""));
	});
	*/

	// 入力データ作成 
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

	//学習開始
	std::cout << "model calculate" << std::endl;

	auto lda = sigtm::LDA::makeInstance(TopicNum, inputdata);

	lda->update(IterationNum);

	lda->save(sigtm::LDA::Distribution::DOCUMENT, out_folder);
	lda->save(sigtm::LDA::Distribution::TOPIC, out_folder);
	lda->save(sigtm::LDA::Distribution::TERM_SCORE, out_folder);

	// LDA後の人物間類似度測定
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
	[サンプルデータ]
	 2013年8月後半の人物名をクエリとして収集したtweet (対象人物はlist.txtに記載)
	
	・文書数(人物数)：77
	・ユニーク単語数：80296
	・総単語数：2048595
	*/
	
	std::wstring fpass = L"test data";
	
	auto simirality = Experiment(fpass + L"/src_documents", fpass, false);

	return 0;
}