#include "lib/LDA/lda.h"
#include "SigUtil/lib/file.hpp"

const int TopicNum = 20;
const int IterationNum = 5;

static const std::wregex url_reg(L"http(s)?://([\\w-]+\\.)+[\\w-]+(/[\\w- ./?%&=]*)?");
static const std::wregex htag_reg(L"#(\\w)+");
static const std::wregex res_reg(L"@(\\w)+");
static const std::wregex noise_reg(L"^[ＴWＷwｗω・･、。*＊:：;；ー－…´`ﾟo｡.,_|│~~\\-\\^\"”'’＂@!！?？#⇒() () ｢」{}\\[\\]\/ 　]+$");
static const std::wregex a_hira_kata_reg(L"^[ぁ-んァ-ン0-9０-９]$");

std::vector<std::vector<double>> Experiment(std::wstring src_folder, std::wstring out_folder, bool make_new)
{
	using namespace std;
	using sig::uint;

#if USE_SIGNLP
	// テキストからデータセットを作成する際に使用するフィルタ
	sigtm::FilterSetting filter(true);

	// 使用品詞の設定
	filter.addWordClass(signlp::WordClass::名詞);
	filter.addWordClass(signlp::WordClass::形容詞);
	filter.addWordClass(signlp::WordClass::動詞);
	
	// 形態素解析前のフィルタ処理
	filter.setPreFilter([](wstring& str){
		static auto& replace = sig::ZenHanReplace::GetInstance();

		str = regex_replace(str, url_reg, wstring(L""));
		str = regex_replace(str, htag_reg, wstring(L""));
		str = regex_replace(str, res_reg, wstring(L""));
	});

	// 形態素解析後にフィルタ処理
	filter.setAftFilter([](wstring& str){
		str = regex_replace(str, noise_reg, wstring(L""));
		str = regex_replace(str, a_hira_kata_reg, wstring(L""));
	});
#endif

	// 入力データ作成 
	sigtm::InputDataPtr inputdata;

	if(make_new){
		// 新しくデータセットを作成(現在サポートしているのはテキストからの生成)
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
		// 過去に作成したデータセットを使用 or 自分で指定形式のデータセットを用意する場合
		inputdata = sigtm::InputData::makeInstance(out_folder);
	}

	cout << "model calculate" << endl;

	auto lda = sigtm::LDA::makeInstance(TopicNum, inputdata);
	uint doc_num = lda->getDocumentNum();

	// 学習開始
	lda->update(IterationNum);

	lda->save(sigtm::LDA::Distribution::DOCUMENT, out_folder);
	lda->save(sigtm::LDA::Distribution::TOPIC, out_folder);
	lda->save(sigtm::LDA::Distribution::TERM_SCORE, out_folder);

	// LDA後の人物間類似度測定
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
	[サンプルデータ]
	 2013年8月後半の人物名をクエリとして収集したtweet (対象人物はlist.txtに記載)
	
	・文書数(人物数)：77
	・ユニーク単語数：80296
	・総単語数：2048595
	*/
	
	std::wstring fpass = L"../SigTM/test data";
	
	auto simirality = Experiment(fpass + L"/src_documents", fpass, true);

	return 0;
}