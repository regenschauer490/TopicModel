#include "example.h"
#include "../lib/model/lda_gibbs.h"
#include "../lib/util/compare.hpp"

/*
[ Latent Dirichlet Allocation (estimate by Gibbs Sampling or Collapsed Gibbs Sampling) ]

・標準的なGibbsSamplingによるLDA
・指定するパラメータ
	・K:トピック数
	・iteration：反復回数
	・alpha：トピック分布thetaの平滑化度合．デフォルトは全ての値が 50 / K
	・beta：単語分布phiの平滑化度合．デフォルトは全ての値が 0.01
	・sampling method：Gibbs Sampling か Collapsed Gibbs Sampling
・最終的な反復回数はg_perplexityを参考に模索して決める
・レジューム機能あり
*/

void example_lda_gibbs(InputTextType tt, FilepassString src_folder, FilepassString out_folder, sig::uint num_topics, sig::uint num_iteration, bool resume, bool make_new)
{
	using namespace std;

	auto inputdata = makeInputData(tt, src_folder, out_folder, make_new);

	resume = resume && (!make_new);

	out_folder = sig::modify_dirpass_tail(out_folder, true);
	const FilepassString perp_pass = out_folder + SIG_TO_FPSTR("perplexity_gibbs.txt");
	if (!resume) sig::clear_file(perp_pass);

	auto savePerplexity = [&](sigtm::LDA const* lda)
	{
		double perp = lda->getPerplexity();
		auto split = sig::split(std::to_string(perp), ",");
		sig::save_line(sig::cat(split, ""), perp_pass, sig::WriteMode::append);
	};

	auto lda = sigtm::LDA_Gibbs::makeInstance(num_topics, inputdata, resume);
	// sigtm::LDA_Gibbs::makeInstance<sigtm::LDA_Gibbs::GibbsSampling>(resume, TopicNum, inputdata);
	auto doc_num = lda->getDocumentNum();

	// 学習開始
	cout << "model training" << endl;
	lda->train(num_iteration, savePerplexity);

	lda->save(sigtm::LDA::Distribution::DOCUMENT, out_folder);
	lda->save(sigtm::LDA::Distribution::TOPIC, out_folder);
	lda->save(sigtm::LDA::Distribution::TERM_SCORE, out_folder);

	auto theta = lda->getTheta();

	auto phi = lda->getPhi();
	for (sig::uint i = 0; i<phi.size(); ++i) {
		sig::save_num(phi[i], out_folder + SIG_TO_FPSTR("phi") + sig::to_fpstring(i), "\n");
	}

	// document間の類似度測定
	vector< vector<double> > similarity(doc_num, vector<double>(doc_num, 0));

	for (sig::uint i = 0; i < doc_num; ++i) {
		for (sig::uint j = 0; j < i; ++j) similarity[i][j] = similarity[j][i];
		for (sig::uint j = i; j < doc_num; ++j) {
			similarity[i][j] = sig::fromJust(sigtm::compare<sigtm::LDA::Distribution::DOCUMENT>(lda, i, j).method<sigtm::CompareMethodD::JS_DIV>());
		}
	}
	sigtm::compare<sigtm::LDA::Distribution::TERM_SCORE>(lda, 0, 1).method<sigtm::CompareMethodV::COS>();
	//sig::SaveCSV(similarity, names, names, out_folder + L"similarity_lda.csv");
}
