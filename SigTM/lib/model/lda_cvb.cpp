/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#include "lda_cvb.h"
#include "SigUtil/lib/calculation.hpp"
#include "SigUtil/lib/tools/convergence.hpp"
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
		sig::compound_assignment(sig::assign_plus_t(), omega_[t.self_id], omega_rand);
		sig::compound_assignment(sig::assign_plus_t(), lambda_[t.word_id], omega_rand);
		sig::compound_assignment(sig::assign_plus_t(), gamma_[t.doc_id], omega_rand);
		sig::compound_assignment(sig::assign_plus_t(), topic_sum_, omega_rand);
	}
}

void LDA_CVB0::update(Token const& t)
{
	//トピック比率の更新
	const auto updateTopic = [&](Token const& t)
	{
		double sum = 0;
		for (TopicId k = 0; k < K_; ++k){
			WordId v = t.word_id;
			omega_[t.self_id][k] = (gamma_[t.doc_id][k] + alpha_[k]) * (lambda_[v][k] + beta_[v]) / (topic_sum_[k] + beta_sum_);
		}
		sig::normalize_dist(omega_[t.self_id]);
	};

	sig::compound_assignment(sig::assign_minus_t(), lambda_[t.word_id], omega_[t.self_id]);
	sig::compound_assignment(sig::assign_minus_t(), gamma_[t.doc_id], omega_[t.self_id]);
	sig::compound_assignment(sig::assign_minus_t(), topic_sum_, omega_[t.self_id]);

	updateTopic(t);

	sig::compound_assignment(sig::assign_plus_t(), lambda_[t.word_id], omega_[t.self_id]);
	sig::compound_assignment(sig::assign_plus_t(), gamma_[t.doc_id], omega_[t.self_id]);
	sig::compound_assignment(sig::assign_plus_t(), topic_sum_, omega_[t.self_id]);
}

void LDA_CVB0::train(uint iteration_num, std::function<void(LDA const*)> callback)
{
	const auto iteration_impl = [&]{
		for (uint i = 0; i < iteration_num; ++i, ++total_iter_ct_){
			std::string numstr = "iteration: " + std::to_string(total_iter_ct_ + 1);
			std::cout << numstr << std::endl;
			std::for_each(std::begin(tokens_), std::end(tokens_), std::bind(&LDA_CVB0::update, this, std::placeholders::_1));
			callback(this);
		}
	};

	iteration_impl();
	//saveResumeData();
	calcTermScore(getPhi(), term_score_);
}

void LDA_CVB0::save(Distribution target, FilepassString save_folder, bool detail) const
{
	save_folder = sig::modify_dirpass_tail(save_folder, true);

	switch (target){
	case Distribution::DOCUMENT:
		printTopic(getTheta(), input_data_->doc_names_, save_folder + SIG_TO_FPSTR("document_cvb0"));
		break;
	case Distribution::TOPIC:
		printWord(getPhi(), std::vector<FilepassString>(), input_data_->words_, detail ? nothing : sig::Just<uint>(20), save_folder + SIG_TO_FPSTR("topic_cvb0"));
		break;
	case Distribution::TERM_SCORE:
		printWord(getTermScore(), std::vector<FilepassString>(), input_data_->words_, detail ? nothing : sig::Just<uint>(20), save_folder + SIG_TO_FPSTR("term-score_cvb0"));
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
	return sig::map([&](uint i){ return lambda_[i][k_id] / sum; }, sig::seqn(0, 1, V_));
}


auto LDA_CVB0::getWordOfTopic(Distribution target, uint return_word_num, TopicId k_id) const->std::vector< std::tuple<std::wstring, double>>
{
	std::vector< std::tuple<std::wstring, double> > result;

	std::vector<double> df;
	if (target == Distribution::TOPIC) df = getPhi(k_id);
	else if (target == Distribution::TERM_SCORE) df = getTermScore(k_id);
	else{
		std::cout << "LDA_CVB0::getWordOfTopic : Distributionが無効" << std::endl;
		return result;
	}

	return getTopWords(df, return_word_num, input_data_->words_);
}

auto LDA_CVB0::getWordOfDocument(uint return_word_num, DocumentId d_id) const->std::vector< std::tuple<std::wstring, double>>
{
	std::vector< std::tuple<std::wstring, double> > result;
	auto top_wscore = getTermScoreOfDocument(getTheta(d_id), getTermScore());

	for (uint i = 0; i<return_word_num; ++i) result.push_back(std::make_tuple(*input_data_->words_.getWord(std::get<0>(top_wscore[i])), std::get<1>(top_wscore[i])));

	return std::move(result);
}


}