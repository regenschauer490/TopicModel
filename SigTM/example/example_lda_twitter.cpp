#include "example.h"
#include "../lib/model/twitter_lda.h"

void example_lda_twitter(FilepassString src_folder, FilepassString out_folder, sig::uint topic_num, sig::uint iteration_num, bool resume, bool make_new)
{
	using namespace std;

	auto inputdata = makeInputData(InputTextType::Tweet, src_folder, out_folder, make_new);

	resume = resume && (!make_new);

	out_folder = sig::modify_dirpass_tail(out_folder, true);
	const FilepassString perp_pass = out_folder + SIG_TO_FPSTR("perplexity_twlda.txt");
	if (!resume) sig::clear_file(perp_pass);

	auto savePerplexity = [&](sigtm::TwitterLDA const* lda)
	{
		double perp = lda->getPerplexity();
		auto split = sig::split(std::to_string(perp), ",");
		sig::save_line(sig::cat(split, ""), perp_pass, sig::WriteMode::append);
	};

	auto lda = sigtm::TwitterLDA::makeInstance(resume, topic_num, inputdata);
	
	// 学習開始
	cout << "model training" << endl;
	lda->train(iteration_num, savePerplexity);

	lda->save(sigtm::TwitterLDA::Distribution::USER, out_folder);
	lda->save(sigtm::TwitterLDA::Distribution::TOPIC, out_folder);
	lda->save(sigtm::TwitterLDA::Distribution::TERM_SCORE, out_folder);

	auto theta = lda->getTheta();
	auto phi = lda->getPhi();
	auto phib = lda->getPhiBackground();

	sig::save_num(theta, out_folder + SIG_TO_FPSTR("theta"), ",");
	sig::save_num(phi, out_folder + SIG_TO_FPSTR("phi"), ",");
	sig::save_num(phib, out_folder + SIG_TO_FPSTR("phib"), ",");
}
