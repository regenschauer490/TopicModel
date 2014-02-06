
#include "LDA/lda.h"

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
	using namespace std;
	using namespace signlp;
	using namespace sigdm;

	const uint doc_num = DocumentNum; 


	sigdm::FilterSetting filter(true);

	//使用品詞の設定
	filter.AddWordClass(WordClass::名詞);
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

	// 入力データ作成 
	sigdm::InputDataPtr inputdata;
	vector< vector<double> > similarity(doc_num, vector<double>(doc_num, 0));

	if(make_new){
		auto doc_pass = sig::GetFileNames(src_folder, false);
		assert(doc_pass, "fail to find src documents");
#if USE_MECAB
		vector<vector<wstring>> docs;
		for (auto dp : *doc_pass){
			docs.push_back(sig::STRtoWSTR(*sig::ReadLine<std::string>(sig::DirpassTailModify(src_folder, true) + dp)));
			std::wcout << docs[0][0];
		}
		inputdata = sigdm::InputDataFactory::MakeInstance(docs, filter, out_folder);
#endif
	}
	else{
		inputdata = sigdm::InputDataFactory::MakeInstance(out_folder);
	}

	//学習開始
	std::cout << "model calculate" << std::endl;

	auto lda = sigdm::LDA::MakeInstance(TopicNum, inputdata);

	lda->Update(IterationNum);

	lda->Save(sigdm::LDA::Distribution::DOCUMENT, out_folder);
	lda->Save(sigdm::LDA::Distribution::TOPIC, out_folder);
	lda->Save(sigdm::LDA::Distribution::TERM_SCORE, out_folder);

	// LDA後の人物間類似度測定
	for (uint i = 0; i < doc_num; ++i){
		for (uint j = 0; j < i; ++j)	similarity[i][j] = similarity[j][i];
		for (uint j = i; j < doc_num; ++j)	similarity[i][j] = lda->CompareDistribution(sigdm::CompareMethodD::JS_DIV, sigdm::LDA::Distribution::DOCUMENT, i, j);
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