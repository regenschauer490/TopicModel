#include "lib/LDA/lda_gibbs.h"
#include "SigUtil/lib/file.hpp"

const int TopicNum = 20;
const int IterationNum = 1000;

static const std::wregex url_reg(L"http(s)?://([\\w-]+\\.)+[\\w-]+(/[\\w- ./?%&=]*)?");
static const std::wregex htag_reg(L"#(\\w)+");
static const std::wregex res_reg(L"@(\\w)+");
static const std::wregex noise_reg(L"^[ＴWＷwｗω・･、。*＊:：;；ー－…´`ﾟo｡.,_|│~~\\-\\^\"”'’＂@!！?？#⇒() () ｢」{}\\[\\]\/ 　]+$");
static const std::wregex a_hira_kata_reg(L"^[ぁ-んァ-ン0-9０-９]$");


sigtm::InputDataPtr makeInputData(std::wstring src_folder, std::wstring out_folder, bool make_new)
{
	using namespace std;
	using sig::uint;

#if USE_SIGNLP
	// テキストからデータセットを作成する際に使用するフィルタ
	sigtm::FilterSetting filter(true);

	// 使用品詞の設定
	filter.addWordClass(signlp::WordClass::名詞);
	filter.addWordClass(signlp::WordClass::形容詞);
	//filter.addWordClass(signlp::WordClass::動詞);
	
	// 形態素解析前のフィルタ処理
	filter.setPreFilter([](wstring& str){
		static auto& replace = sig::ZenHanReplace::GetInstance();
		static sig::TagDealer<std::wstring> tag_dealer(L"<", L">");

		auto tmp = tag_dealer.decode(str, L"TEXT");
		str = tmp ? sig::fromJust(tmp) : str;
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
/*		auto doc_pass = sig::get_file_names(, false);

		vector<vector<wstring>> docs;
		for (auto dp : sig::fromJust(doc_pass)){
			auto tdoc = sig::read_line<wstring>(sig::modify_dirpass_tail(src_folder, true) + dp);
			docs.push_back(
				sig::fromJust(tdoc)
			);
		}
*/
		inputdata = sigtm::InputDataFromText::makeInstance(src_folder, filter, out_folder);
#else
		assert(false);
#endif
	}
	else{
		// 過去に作成したデータセットを使用 or 自分で指定形式のデータセットを用意する場合
		inputdata = sigtm::InputData::makeInstance(out_folder);
	}

	return inputdata;
}

void experiment(std::wstring src_folder, std::wstring out_folder, bool resume, bool make_new)
{
	using namespace std;
	using sig::uint;

	auto inputdata = makeInputData(src_folder, out_folder, make_new);

	resume = resume && (!make_new);
	
	const std::wstring perp_pass = sig::modify_dirpass_tail(out_folder, true) + L"perplexity_gibbs.txt";
	if(!resume) sig::clear_file(perp_pass);
	
	auto savePerplexity = [&](sigtm::LDA const* lda)
	{
		double perp = lda->getPerplexity();
		auto split = sig::split(std::to_string(perp), ",");
		sig::save_line(sig::cat_str(split, ""), perp_pass, sig::WriteMode::append);
	};

	auto lda = sigtm::LDA_Gibbs::makeInstance(resume, TopicNum, inputdata);
	uint doc_num = lda->getDocumentNum();

	cout << "model calculate" << endl;

	// 学習開始
	lda->train(IterationNum, savePerplexity);

	lda->save(sigtm::LDA::Distribution::DOCUMENT, out_folder);
	lda->save(sigtm::LDA::Distribution::TOPIC, out_folder);
	lda->save(sigtm::LDA::Distribution::TERM_SCORE, out_folder);

	// LDA後の人物間類似度測定
	vector< vector<double> > similarity(doc_num, vector<double>(doc_num, 0));

	for (uint i = 0; i < doc_num; ++i){
		for (uint j = 0; j < i; ++j)	similarity[i][j] = similarity[j][i];
		for (uint j = i; j < doc_num; ++j)similarity[i][j] = sig::fromJust(sigtm::compare<sigtm::LDA::Distribution::DOCUMENT>(lda, i, j).method(sigtm::CompareMethodD::JS_DIV));
		//lda->compareDistribution(sigtm::CompareMethodD::JS_DIV, sigtm::LDA::Distribution::DOCUMENT, i, j);
	}
	
	//sig::SaveCSV(similarity, names, names, out_folder + L"similarity_lda.csv");
}

#include "lib/LDA/mrlda.h"

void experiment2(std::wstring src_folder, std::wstring out_folder, bool resume, bool make_new)
{
	auto inputdata = makeInputData(src_folder, out_folder, make_new);
	
	resume = resume && (!make_new);
	std::cout << resume << " " << make_new;
	const std::wstring perp_pass = sig::modify_dirpass_tail(out_folder, true) + L"perplexity_mrlda.txt";
	if(!resume) sig::clear_file(perp_pass);

	sig::TimeWatch tw;
	auto savePerplexity = [&](sigtm::LDA const* lda)
	{		
		tw.save();
		std::cout << "1 iteration time: " << tw.get_total_time() << std::endl;
		tw.reset();

		double perp = lda->getPerplexity();
		auto split = sig::split(std::to_string(perp), ",");
		sig::save_line(sig::cat_str(split, ""), perp_pass, sig::WriteMode::append);

		tw.restart();
	};

	auto mrlda = sigtm::MrLDA::makeInstance(resume, TopicNum, inputdata);
	mrlda->train(100, savePerplexity);
}

int main()
{
	/*
	[サンプルデータ]
	
	
	・文書数(人物数)：
	・ユニーク単語数：
	・総単語数：
	*/
	
	std::wstring data_folder_pass = L"../SigTM/test data";
	std::wstring input_text_pass = data_folder_pass + L"/processed";
	
	experiment(input_text_pass, data_folder_pass, true, false);

	return 0;
}