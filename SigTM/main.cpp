#include "lib/helper/input_text.h"
#include "SigUtil/lib/file.hpp"

const int TopicNum = 20;
const int IterationNum = 100;

static const std::wregex url_reg(L"http(s)?://([\\w-]+\\.)+[\\w-]+(/[\\w- ./?%&=]*)?");
static const std::wregex htag_reg(L"#(\\w)+");
static const std::wregex res_reg(L"@(\\w)+");
static const std::wregex noise_reg(L"^[ＴWＷwｗω・･、。*＊:：;；ー－…´`ﾟo｡.,_|│~~\\-\\^\"”'’＂@!！?？#⇒() () ｢」{}\\[\\]\/ 　]+$");
static const std::wregex a_hira_kata_reg(L"^[ぁ-んァ-ン0-9０-９]$");

/*
	[ 入力形式のデータ作成 ]

・新規作成(外部ファイル or プログラム内の変数)
	・別途MeCabのインストールが必要

・過去の作成データを使用
	・tokenデータ：テキスト中の各トークンに関する情報
	・vocabデータ：出現単語に関する情報
*/
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
#if USE_SIGNLP
		// 新しくデータセットを作成(外部ファイルから生成)
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


/*
	[ Latent Dirichlet Allocation (estimate by Gibbs Sampling or Collapsed Gibbs Sampling) ]

・標準的なGibbsSamplingによるLDA
・指定するパラメータ
	・K:トピック数
	・iteration：反復回数
	・alpha：トピック分布thetaの平滑化度合．デフォルトは全ての値が 50 / K
	・beta：単語分布phiの平滑化度合．デフォルトは全ての値が 0.01
	・sampling method：Gibbs Sampling か Collapsed Gibbs Sampling
・最終的な反復回数はperplexityを参考に模索して決める
・レジューム機能あり
*/
#include "lib/model/lda_gibbs.h"

void sample1(std::wstring src_folder, std::wstring out_folder, bool resume, bool make_new)
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
	// sigtm::LDA_Gibbs::makeInstance<sigtm::LDA_Gibbs::GibbsSampling>(resume, TopicNum, inputdata);
	uint doc_num = lda->getDocumentNum();

	cout << "model calculate" << endl;

	// 学習開始
	lda->train(IterationNum, savePerplexity);

	lda->save(sigtm::LDA::Distribution::DOCUMENT, out_folder);
	lda->save(sigtm::LDA::Distribution::TOPIC, out_folder);
	lda->save(sigtm::LDA::Distribution::TERM_SCORE, out_folder);

	auto theta = lda->getTheta();
	auto theta1 = lda->getTheta(1);

	// document間の類似度測定
	vector< vector<double> > similarity(doc_num, vector<double>(doc_num, 0));

	for (uint i = 0; i < doc_num; ++i){
		for (uint j = 0; j < i; ++j) similarity[i][j] = similarity[j][i];
		for (uint j = i; j < doc_num; ++j){
			similarity[i][j] = sig::fromJust(sigtm::compare<sigtm::LDA::Distribution::DOCUMENT>(lda, i, j).method(sigtm::CompareMethodD::JS_DIV));
		}
	}
	//sig::SaveCSV(similarity, names, names, out_folder + L"similarity_lda.csv");
}


/*
	[ Latent Dirichlet Allocation (estimate by Zero-Order Collapsed Variational Bayesian inference) ]

・標準的なCVB0によるLDA
・指定するパラメータ
	・K : トピック数
	・iteration：反復回数
	・alpha：トピック分布thetaの平滑化度合．デフォルトは全ての値が 50 / K
	・beta：単語分布phiの平滑化度合．デフォルトは全ての値が 0.01
・最終的な反復回数はperplexityを参考に模索して決める
 */
#include "lib/model/lda_cvb.h"

void sample2(std::wstring src_folder, std::wstring out_folder, bool resume, bool make_new)
{
	using namespace std;
	using sig::uint;

	auto inputdata = makeInputData(src_folder, out_folder, make_new);

	resume = resume && (!make_new);

	const std::wstring perp_pass = sig::modify_dirpass_tail(out_folder, true) + L"perplexity_cvb.txt";
	if (!resume) sig::clear_file(perp_pass);

	auto savePerplexity = [&](sigtm::LDA const* lda)
	{
		double perp = lda->getPerplexity();
		auto split = sig::split(std::to_string(perp), ",");
		sig::save_line(sig::cat_str(split, ""), perp_pass, sig::WriteMode::append);
	};

	auto lda = sigtm::LDA_CVB0::makeInstance(resume, TopicNum, inputdata);
	uint doc_num = lda->getDocumentNum();

	cout << "model calculate" << endl;

	// 学習開始
	lda->train(IterationNum, savePerplexity);

	lda->save(sigtm::LDA::Distribution::DOCUMENT, out_folder);
	lda->save(sigtm::LDA::Distribution::TOPIC, out_folder);
	lda->save(sigtm::LDA::Distribution::TERM_SCORE, out_folder);
}


#include "lib/model/mrlda.h"

void sample3(std::wstring src_folder, std::wstring out_folder, bool resume, bool make_new)
{
	auto inputdata = makeInputData(src_folder, out_folder, make_new);
	
	resume = resume && (!make_new);
	std::cout << resume << " " << make_new;

	const std::wstring perp_pass = sig::modify_dirpass_tail(out_folder, true) + L"perplexity_mrlda.txt";
	const std::wstring time_pass = sig::modify_dirpass_tail(out_folder, true) + L"time_mrlda.txt";
	if(!resume){
		sig::clear_file(perp_pass);
		sig::clear_file(time_pass);
	}

	sig::TimeWatch tw;
	auto savePerplexity = [&](sigtm::LDA const* lda)
	{		
		tw.save();
		std::cout << "1 iteration time: " << tw.get_total_time() << std::endl;
		sig::save_line(tw.get_total_time<std::chrono::seconds>(), time_pass, sig::WriteMode::append);
		tw.reset();

		double perp = lda->getPerplexity();
		auto split = sig::split(std::to_string(perp), ",");
		sig::save_line(sig::cat_str(split, ""), perp_pass, sig::WriteMode::append);

		tw.restart();
	};

	auto mrlda = sigtm::MrLDA::makeInstance(resume, TopicNum, inputdata);
	mrlda->train(100, savePerplexity);
}

/*
#include "lib/model/twitter_lda.h"

void sample4(std::wstring src_folder, std::wstring out_folder, bool resume, bool make_new)
{
	using namespace std;
	using sig::uint;

	auto inputdata = makeInputData(src_folder, out_folder, make_new);

	resume = resume && (!make_new);

	const std::wstring perp_pass = sig::modify_dirpass_tail(out_folder, true) + L"perplexity_twlda.txt";
	if (!resume) sig::clear_file(perp_pass);

	auto savePerplexity = [&](sigtm::LDA const* lda)
	{
		double perp = lda->getPerplexity();
		auto split = sig::split(std::to_string(perp), ",");
		sig::save_line(sig::cat_str(split, ""), perp_pass, sig::WriteMode::append);
	};

	auto lda = sigtm::TwitterLDA::makeInstance(resume, TopicNum, inputdata);
	uint doc_num = lda->getDocumentNum();

	cout << "model calculate" << endl;

	// 学習開始
	lda->train(IterationNum, savePerplexity);

	lda->save(sigtm::LDA::Distribution::DOCUMENT, out_folder);
	lda->save(sigtm::LDA::Distribution::TOPIC, out_folder);
	lda->save(sigtm::LDA::Distribution::TERM_SCORE, out_folder);
}
*/

#include "lib/helper/SigNLP/polar_spin.hpp"
#include "lib/helper/SigNLP/mecab_wrapper.hpp"
void polar_train()
{	
	using Spin = signlp::SpinModel;
	using signlp::WordClass;
	auto& mecab = signlp::MecabWrapper::getInstance();
	auto texts = sig::read_line<std::string>(L"Z:/Nishimura/IVRC/data/streaming/test.txt");
	auto label_texts = sig::read_line<std::string>(L"Z:/Nishimura/IVRC/evaluation_lib/polar.csv");

	Spin::Graph graph;
	std::unordered_map<std::wstring, Spin::pNode> wmap;
	const int sigma2 = 9;
	const double z = std::sqrt(2 * 3.1415 * sigma2);

	for (auto const& line : sig::fromJust(label_texts)){
		auto sp = sig::split(line, ",");
		auto w = sig::str_to_wstr(sp[0]);
		auto sc = std::stod(sp[3]);
		if (std::abs(sc) < 0.75) continue;

		wmap.emplace(w, Spin::make_node(graph, w, sc));
	}

	for (auto const& line : sig::fromJust(texts)){
		auto parsed = sig::str_to_wstr(
			mecab.parseGenkeiThroughFilter(line, [](WordClass wc){ return WordClass::名詞 == wc || WordClass::形容詞 == wc || WordClass::動詞 == wc; })
		);
		
		// 新規ノード(単語)追加
		for (auto const& w : parsed){
			if (!wmap.count(w)){
				wmap.emplace(w, Spin::make_node(graph, w));
			}
		}

		// エッジ追加
		for (int i=0; i<parsed.size(); ++i){
			auto v = wmap[parsed[i]];
			/* 左側の単語
			for (int l=i-1, dist=0; l>=0; --l, ++dist){
				double weight = z * std::exp(-0.5 * std::pow((l - i) / sigma2, 2));
				auto e = boost::edge(v, wmap[parsed[l]], graph);

				if (e.second){
					boost::get(boost::edge_weight, graph, e.first) += weight;
				}
				else{
					Spin::make_edge(graph, v, wmap[parsed[l]], weight);
				}
			}*/
			// 右側の単語
			for (int r = i + 1, dist = 0; r<parsed.size(); ++r, ++dist){
				auto v2 = wmap[parsed[r]];
				auto weight = 1 * std::exp(-std::pow((r - i), 2) / (2*sigma2));
				auto e = boost::edge(v, v2, graph);

				if (e.second){
					boost::get(boost::edge_weight, graph, e.first) += weight;
				}
				else{
					Spin::make_edge(graph, v, v2, weight);
				}				
			}
		}
	}
	
	for (int i=1; i<21; ++i){
		signlp::SpinModel spin(graph, 10, 0.1 * i);

		auto callback = [&](signlp::SpinModel const* obj){
			sig::save_line(obj->getErrorRate(), L"./test data/spinmodel/error" + std::to_wstring(i) + L".txt", sig::WriteMode::append);
			sig::save_line(obj->getMeanPolar(), L"./test data/spinmodel/mp" + std::to_wstring(i) + L".txt", sig::WriteMode::append);
		};

		spin.train(50, callback);
		auto score = spin.getScore();
		for (auto e : score) sig::save_line(e.first + L"," + std::to_wstring(e.second), L"./test data/spinmodel/score" + std::to_wstring(i) + L".txt", sig::WriteMode::append);
	}

	getchar();
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

	setlocale(LC_ALL, "Japanese");

	polar_train();
	//sample3(input_text_pass, data_folder_pass, true, false);

	return 0;
}