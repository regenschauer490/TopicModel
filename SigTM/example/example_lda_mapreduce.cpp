#include "example.h"
#include "../lib/model/mrlda.h"
#include "SigUtil/lib/tools/time_watch.hpp"

#if SIG_MSVC_ENV

void example_lda_mapreduce(InputTextType tt, FilepassString src_folder, FilepassString out_folder, sig::uint num_topics, sig::uint num_iteration, bool resume, bool make_new)
{
	using namespace std;

	auto inputdata = makeInputData(tt, src_folder, out_folder, make_new);

	resume = resume && (!make_new);

	const wstring perp_pass = sig::modify_dirpath_tail(out_folder, true) + L"perplexity_mrlda.txt";
	const wstring time_pass = sig::modify_dirpath_tail(out_folder, true) + L"time_mrlda.txt";
	if (!resume) {
		sig::clear_file(perp_pass);
		sig::clear_file(time_pass);
	}

	sig::TimeWatch<> tw;
	auto savePerplexity = [&](sigtm::LDA const* lda)
	{
		tw.save();
		cout << "1 iteration time: " << tw.get_total_time() << endl;
		sig::save_line(tw.get_total_time<chrono::seconds>(), time_pass, sig::WriteMode::append);
		tw.reset();

		double perp = lda->getPerplexity();
		auto split = sig::split(to_string(perp), ",");
		sig::save_line(sig::cat(split, ""), perp_pass, sig::WriteMode::append);

		tw.restart();
	};

	auto mrlda = sigtm::MrLDA::makeInstance(num_topics, inputdata, resume);

	// 学習開始
	cout << "model training" << endl;
	mrlda->train(num_iteration, savePerplexity);
}

#endif
