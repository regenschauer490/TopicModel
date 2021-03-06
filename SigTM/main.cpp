﻿#include "SigUtil/lib/file.hpp"
#include "SigUtil/lib/modify/remove.hpp"
#include "SigUtil/lib/functional/list_deal.hpp"
#include "SigUtil/lib/tools/tag_dealer.hpp"
#include "SigUtil/lib/tools/time_watch.hpp"

const int TopicNum = 30;
const int IterationNum = 100;

// 入力テキストの種類 (Webページやレビュー文などの各記事はDocument, マイクロブログでの各ユーザの投稿はTweet)
enum class InputTextType { Document, Tweet };

static const std::wregex url_reg(L"http(s)?://([\\w-]+\\.)+[\\w-]+(/[\\w- ./?%&=]*)?");
static const std::wregex htag_reg(L"#(\\w)+");
static const std::wregex res_reg(L"@(\\w)+");
static const std::wregex noise_reg(L"^[ＴWＷwｗω・･、。*＊:：;；ー－…´`ﾟo｡.,_|│~~\\-\\^\"”'’＂@!！?？#⇒() () ｢」{}\\[\\]\\/ 　]+$");
static const std::wregex a_hira_kata_reg(L"^[ぁ-んァ-ン0-9０-９]$");


#include "lib/sigtm.hpp"

#if USE_SIGNLP
#include "lib/helper/document_loader_text.hpp"
#else
#include "lib/helper/document_loader.hpp"
#endif

/*
	[ 入力形式のデータ作成 ]

・新規作成(外部ファイル or プログラム内の変数)
	・別途MeCabのインストールが必要

・過去の作成データを使用
	・tokenデータ：テキスト中の各トークンに関する情報
	・vocabデータ：出現単語に関する情報
*/
sigtm::DocumentSetPtr makeInputData(InputTextType tt, std::wstring src_folder, std::wstring out_folder, bool make_new)
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
		static auto& replace = sig::ZenHanReplace::get_instance();
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
	sigtm::DocumentSetPtr inputdata;

	if(make_new){
#if USE_SIGNLP
		// 新しくデータセットを作成(外部ファイルから生成)
		if (InputTextType::Tweet == tt) inputdata = sigtm::DocumentLoaderFromText::makeInstanceFromTweet(src_folder, filter, out_folder);
		else inputdata = sigtm::DocumentLoaderFromText::makeInstance(src_folder, filter, out_folder);
#else
		assert(false);
#endif
	}
	else{
		// 過去に作成したデータセットを使用 or 自分で指定形式のデータセットを用意する場合
		inputdata = sigtm::DocumentLoader::makeInstance(out_folder);
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

void sample1(InputTextType tt, std::wstring src_folder, std::wstring out_folder, bool resume, bool make_new)
{
	using namespace std;
	using sig::uint;

	auto inputdata = makeInputData(tt, src_folder, out_folder, make_new);

	resume = resume && (!make_new);
	
	out_folder = sig::modify_dirpass_tail(out_folder, true);
	const std::wstring perp_pass = out_folder + L"perplexity_gibbs.txt";
	if(!resume) sig::clear_file(perp_pass);
	
	auto savePerplexity = [&](sigtm::LDA const* lda)
	{
		double perp = lda->getPerplexity();
		auto split = sig::split(std::to_string(perp), ",");
		sig::save_line(sig::cat(split, ""), perp_pass, sig::WriteMode::append);
	};

	auto lda = sigtm::LDA_Gibbs::makeInstance(resume, TopicNum, inputdata);
	// sigtm::LDA_Gibbs::makeInstance<sigtm::LDA_Gibbs::GibbsSampling>(resume, TopicNum, inputdata);
	uint doc_num = lda->getDocumentNum();

	cout << "model calculate" << endl;

	// 学習開始
	lda->train(100, savePerplexity);

	lda->save(sigtm::LDA::Distribution::DOCUMENT, out_folder);
	lda->save(sigtm::LDA::Distribution::TOPIC, out_folder);
	lda->save(sigtm::LDA::Distribution::TERM_SCORE, out_folder);

	auto theta = lda->getTheta();
	
	auto phi = lda->getPhi();
	for (uint i=0; i<phi.size(); ++i){
		sig::save_num(phi[i], out_folder + L"tphi/phi" + std::to_wstring(i), "\n");
	}

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

void sample2(InputTextType tt, std::wstring src_folder, std::wstring out_folder, bool resume, bool make_new)
{
	using namespace std;
	using sig::uint;

	auto inputdata = makeInputData(tt, src_folder, out_folder, make_new);

	resume = resume && (!make_new);

	const std::wstring perp_pass = sig::modify_dirpass_tail(out_folder, true) + L"perplexity_cvb.txt";
	if (!resume) sig::clear_file(perp_pass);

	auto savePerplexity = [&](sigtm::LDA const* lda)
	{
		double perp = lda->getPerplexity();
		auto split = sig::split(std::to_string(perp), ",");
		sig::save_line(sig::cat(split, ""), perp_pass, sig::WriteMode::append);
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

void sample3(InputTextType tt, std::wstring src_folder, std::wstring out_folder, bool resume, bool make_new)
{
	auto inputdata = makeInputData(tt, src_folder, out_folder, make_new);
	
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
		sig::save_line(sig::cat(split, ""), perp_pass, sig::WriteMode::append);

		tw.restart();
	};

	auto mrlda = sigtm::MrLDA::makeInstance(resume, TopicNum, inputdata);
	mrlda->train(100, savePerplexity);
}


#include "lib/model/twitter_lda.h"

void sample4(std::wstring src_folder, std::wstring out_folder, bool resume, bool make_new)
{
	using namespace std;
	using sig::uint;

	auto inputdata = makeInputData(InputTextType::Tweet, src_folder, out_folder, make_new);

	resume = resume && (!make_new);

	out_folder = sig::modify_dirpass_tail(out_folder, true);
	const std::wstring perp_pass = out_folder + L"perplexity_twlda.txt";
	if (!resume) sig::clear_file(perp_pass);

	auto savePerplexity = [&](sigtm::TwitterLDA const* lda)
	{
		double perp = lda->getPerplexity();
		auto split = sig::split(std::to_string(perp), ",");
		sig::save_line(sig::cat(split, ""), perp_pass, sig::WriteMode::append);
	};

	auto lda = sigtm::TwitterLDA::makeInstance(resume, 30, inputdata);

	cout << "model calculate" << endl;

	// 学習開始
	lda->train(70, savePerplexity);

	lda->save(sigtm::TwitterLDA::Distribution::USER, out_folder);
	lda->save(sigtm::TwitterLDA::Distribution::TOPIC, out_folder);
	lda->save(sigtm::TwitterLDA::Distribution::TERM_SCORE, out_folder);

	auto theta = lda->getTheta();
	auto phi = lda->getPhi();
	auto phib = lda->getPhiBackground();

	sig::save_num(theta, out_folder + L"theta", ",");
	sig::save_num(phi, out_folder + L"phi", ",");
	sig::save_num(phib, out_folder + L"phib", ",");

	getchar();
}


#include "lib/model/ctr.h"
#include "lib/helper/ctr_validation.hpp"

void sample5(std::wstring src_folder, std::wstring out_folder, bool resume, bool make_new)
{
	using namespace std;
	using sig::uint;

	bool for_user_recommend = true;

	resume = resume && (!make_new);

	out_folder = sig::modify_dirpass_tail(out_folder, true);
	const std::wstring perp_pass = out_folder + L"perplexity_twlda.txt";
	if (!resume) sig::clear_file(perp_pass);

	// file load

	sigtm::DocumentLoader::PF corpus_parser = [&](sigtm::TokenList& tokens, sigtm::WordSet& words)
	{
		sigtm::DocumentLoaderSetInfo info;

		auto token_file = *sig::load_line(out_folder + L"token");
		auto vocab_file = *sig::load_line<std::wstring>(out_folder + L"vocab");

		uint total_ct = 0, did = 0;
		for(auto const& line : token_file){
			auto parsed = sig::split(line, " ");

			for (uint n = 1, length = std::stoul(parsed[0])+1; n < length; ++n) {
				auto word_count = sig::split(parsed[n], ":");
				uint wid = std::stoul(word_count[0]);

				for (uint m = 0, wct = std::stoul(word_count[1]); m < wct; ++m){
					tokens.push_back(sigtm::Token(total_ct, did, wid));
					++total_ct;
				}
				words.emplace(wid, vocab_file[wid]);
			}
			++did;
			info.doc_names_.push_back(L"doc " + std::to_wstring(did));
		}

		info.doc_num_ = did;
		info.doc_type_ = sigtm::DocumentType::Defaut;
		info.is_token_sorted_ = true;
		info.working_directory_ = out_folder;

		return info;
	};

	auto docs = make_new
		? makeInputData(InputTextType::Document, src_folder, out_folder, make_new)
		: sigtm::DocumentLoader::makeInstance(corpus_parser);
	
	/*
	auto lda = sigtm::LDA_Gibbs::makeInstance(false, 20, docs);
	auto savePerplexity = [&](sigtm::LDA const* lda)
	{
		double perp = lda->getPerplexity();
		auto split = sig::split(std::to_string(perp), ",");
		std::cout << sig::cat(split, "") << std::endl;
	};
	lda->train(100, savePerplexity);
	auto theta = lda->getTheta();
	auto phi = lda->getPhi();
	sig::save_num(theta, out_folder + L"theta", " ");
	sig::save_num(phi, out_folder + L"phi", " ");
	*/
	
	auto user_ratings = *sig::load_num2d<uint>(out_folder + L"user", " ");	
	for (auto& vec : user_ratings) vec = sig::drop(1, std::move(vec));
	//auto item_ratings = *sig::load_num2d<uint>(out_folder + L"item", " ");
	//for (auto& vec : item_ratings) sig::drop(1, vec);

	auto ratings = sigtm::SparseBooleanMatrix::makeInstance(user_ratings, true);

	auto hparam = sigtm::CtrHyperparameter::makeInstance(true);

	if (auto theta = sig::load_num2d<double>(out_folder + L"theta", " ")){
		hparam->setTheta(*theta);
	}
	if (auto beta = sig::load_num2d<double>(out_folder + L"beta", " ")){
		hparam->setTheta(*beta);
	}
	
/*	auto ctr = sigtm::CTR::makeInstance(20, hparam, docs, ratings);

	cout << "model calculate" << endl;

	// 学習開始
	ctr->train(500, 10, 10);
*/


	sigtm::CrossValidation<sigtm::CTR> validation(4, for_user_recommend, 20, hparam, docs, ratings, 100, 10, 5);
		
/*	auto tmp_u = *sig::load_num2d<double>(out_folder + L"final_U.dat", " ");
	auto tmp_v = *sig::load_num2d<double>(out_folder + L"final_V.dat", " ");

	validation.debug_set_u(tmp_u);
	validation.debug_set_v(tmp_v);
*/
	{
	auto precision = validation.run(sigtm::Precision<sigtm::CTR>(50, 0.5));
	auto recall = validation.run(sigtm::Recall<sigtm::CTR>(50, 0.5));
	auto fmeasure = sig::zipWith(sigtm::F_MeasureBase(), precision, recall);
	
	sig::save_num(precision, L"./precision.txt", "\n");
	sig::save_num(recall, L"./recall.txt", "\n");
	sig::save_num(fmeasure, L"./f_measure.txt", "\n");
	}
	{
	auto precision = validation.run(sigtm::Precision<sigtm::CTR>(10, sigtm::nothing));
	auto recall = validation.run(sigtm::Recall<sigtm::CTR>(10, sigtm::nothing));
	auto fmeasure = sig::zipWith(sigtm::F_MeasureBase(), precision, recall);

	sig::save_num(precision, L"./precision2.txt", "\n");
	sig::save_num(recall, L"./recall2.txt", "\n");
	sig::save_num(fmeasure, L"./f_measure2.txt", "\n");
	}
	{
	auto precision = validation.run(sigtm::Precision<sigtm::CTR>(100, sigtm::nothing));
	auto recall = validation.run(sigtm::Recall<sigtm::CTR>(100, sigtm::nothing));
	auto fmeasure = sig::zipWith(sigtm::F_MeasureBase(), precision, recall);

	sig::save_num(precision, L"./precision3.txt", "\n");
	sig::save_num(recall, L"./recall3.txt", "\n");
	sig::save_num(fmeasure, L"./f_measure3.txt", "\n");
	}
	{
	auto precision = validation.run(sigtm::Precision<sigtm::CTR>(1000, sigtm::nothing));
	auto recall = validation.run(sigtm::Recall<sigtm::CTR>(1000, sigtm::nothing));
	auto fmeasure = sig::zipWith(sigtm::F_MeasureBase(), precision, recall);

	sig::save_num(precision, L"./precision4.txt", "\n");
	sig::save_num(recall, L"./recall4.txt", "\n");
	sig::save_num(fmeasure, L"./f_measure4.txt", "\n");
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
	
	std::wstring data_folder_pass = L"../../SigTM/test_data";
	std::wstring input_text_pass = data_folder_pass + L"/dataset/document";
	//std::wstring input_tw_pass = data_folder_pass + L"/dataset/tweet";

	setlocale(LC_ALL, "Japanese");
	
	//sample1(InputTextType::Tweet, input_text_pass, data_folder_pass, false, false);
	//sample4(input_tw_pass, data_folder_pass, false, false);
	sample5(data_folder_pass + L"/ctr", data_folder_pass + L"/ctr", false, false);

	return 0;
}