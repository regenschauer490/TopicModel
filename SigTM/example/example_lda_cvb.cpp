#include "example.h"
#include "../lib/model/lda_cvb.h"

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

void example_lda_cvb(InputTextType tt, std::wstring src_folder, std::wstring out_folder, sig::uint topic_num, sig::uint iteration_num, bool resume, bool make_new)
{
	using namespace std;

	auto inputdata = makeInputData(tt, src_folder, out_folder, make_new);

	resume = resume && (!make_new);

	const wstring perp_pass = sig::modify_dirpass_tail(out_folder, true) + L"perplexity_cvb.txt";
	if (!resume) sig::clear_file(perp_pass);

	auto savePerplexity = [&](sigtm::LDA const* lda)
	{
		double perp = lda->getPerplexity();
		auto split = sig::split(std::to_string(perp), ",");
		sig::save_line(sig::cat(split, ""), perp_pass, sig::WriteMode::append);
	};

	auto lda = sigtm::LDA_CVB0::makeInstance(resume, topic_num, inputdata);
	auto doc_num = lda->getDocumentNum();
	
	// 学習開始
	cout << "model training" << endl;
	lda->train(iteration_num, savePerplexity);

	lda->save(sigtm::LDA::Distribution::DOCUMENT, out_folder);
	lda->save(sigtm::LDA::Distribution::TOPIC, out_folder);
	lda->save(sigtm::LDA::Distribution::TERM_SCORE, out_folder);
}
