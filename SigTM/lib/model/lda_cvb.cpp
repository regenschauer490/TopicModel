/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#include "lda_cvb.h"
#include "SigUtil/lib/calculation.hpp"
#include "SigUtil/lib/iteration.hpp"
#include "SigUtil/lib/error_convergence.hpp"
#include "SigUtil/lib/file.hpp"

namespace sigtm
{

void LDA_CVB0::init(bool resume)
{
	int i = -1;
	lambda_ = SIG_INIT_MATRIX(double, V, K, 0);
	gamma_ = SIG_INIT_MATRIX(double, D, K, 0);
	topic_sum_ = SIG_INIT_VECTOR(double, K, 0);

	for (auto const& t : tokens_){
		auto omega_rand = makeRandomDistribution<VectorK<double>>(K_);
		sig::compound_assignment(sig::assign_plus<double>(), lambda_[t.word_id], omega_rand);
		sig::compound_assignment(sig::assign_plus<double>(), gamma_[t.doc_id], omega_rand);
		sig::compound_assignment(sig::assign_plus<double>(), topic_sum_, omega_rand);
	}
}

void LDA_CVB0::update(Token const& t)
{
	//トピック比率の更新
	const auto updateTopic = [&](Token const& t)
	{
		for (TopicId k = 0; k < K_; ++k){
			WordId v = t.word_id;
			omega_[t.self_id][k] = (gamma_[t.doc_id][k] + alpha_[k]) * (lambda_[v][k] + beta_[v]) / (topic_sum_[k] + V_ * beta_[v]);
		}
	};

	sig::compound_assignment(sig::assign_minus<double>(), lambda_[t.word_id], omega_[t.self_id]);
	sig::compound_assignment(sig::assign_minus<double>(), gamma_[t.doc_id], omega_[t.self_id]);
	sig::compound_assignment(sig::assign_minus<double>(), topic_sum_, omega_[t.self_id]);

	updateTopic(t);

	sig::compound_assignment(sig::assign_plus<double>(), lambda_[t.word_id], omega_[t.self_id]);
	sig::compound_assignment(sig::assign_plus<double>(), gamma_[t.doc_id], omega_[t.self_id]);
	sig::compound_assignment(sig::assign_plus<double>(), topic_sum_, omega_[t.self_id]);
}


void LDA_CVB0::save(Distribution target, FilepassString save_folder, bool detail) const
{
	save_folder = sig::modify_dirpass_tail(save_folder, true);

	switch (target){
	case Distribution::DOCUMENT:
		printTopic(getTheta(), input_data_->doc_names_, save_folder + SIG_STR_TO_FPSTR("document_cvb0"));
		break;
	case Distribution::TOPIC:
		printWord(getPhi(), std::vector<FilepassString>(), input_data_->words_, sig::maybe<uint>(20), save_folder + SIG_STR_TO_FPSTR("topic_cvb0"), detail);
		break;
	case Distribution::TERM_SCORE:
		printWord(getTermScore(), std::vector<FilepassString>(), input_data_->words_, sig::maybe<uint>(20), save_folder + SIG_STR_TO_FPSTR("term-score_cvb0"), detail);
		break;
	default:
		std::cout << "LDA_CVB0::save error" << std::endl;
		getchar();
	}
}

auto LDA_CVB0::getTheta() const->MatrixDK<double>
{
	MatrixDK<double> theta;

	for (DocumentId d = 0; d < D_; ++d) theta.push_back(getTheta(d));

	return theta;
}

auto LDA_CVB0::getTheta(DocumentId d_id) const->VectorK<double>
{
	double sum = sig::sum(gamma_[d_id]);
	return sig::map([sum](double v){ return v / sum; }, gamma_[d_id]);
}

auto LDA_CVB0::getPhi() const->MatrixKV<double>
{
	MatrixKV<double> phi;

	for (TopicId k = 0; k < K_; ++k) phi.push_back(getPhi(k));

	return std::move(phi);
}

auto LDA_CVB0::getPhi(TopicId k_id) const->VectorV<double>
{
	double sum = sig::sum_col(lambda_, k_id);
	// computed from the variational distribution
	return sig::map([&](uint i){ return lambda_[i][k_id] / sum; }, sig::seq(0, 1, V_));
}

}