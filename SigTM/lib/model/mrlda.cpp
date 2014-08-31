/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

//#define _SCL_SECURE_NO_WARNINGS

#include "mrlda.h"
#include "../helper/mapreduce_module.h"
#include "SigUtil/lib/calculation.hpp"
#include "SigUtil/lib/iteration.hpp"
#include "SigUtil/lib/distance/norm.hpp"
#include "SigUtil/lib/error_convergence.hpp"
#include "SigUtil/lib/blas.hpp"
#include "SigUtil/lib/file.hpp"
#include <boost/numeric/ublas/io.hpp>

namespace sigtm
{
const uint MrLDA:: max_local_iteration_ = 100;
const double MrLDA::global_convergence_threshold = 0.0001;
const double alpha_convergence_threshold = 0.001;
const uint max_alpha_update_iteration = 100;

const auto resume_info_fname = SIG_STR_TO_FPSTR("mrlda_info");
const auto resume_alpha_fname = SIG_STR_TO_FPSTR("mrlda_alpha");
const auto resume_gamma_fname = SIG_STR_TO_FPSTR("mrlda_gamma");
const auto resume_phi_fname = SIG_STR_TO_FPSTR("mrlda_phi");

void MrLDA::MapTask::process(value_type const& value, VectorK<double>& gamma, MatrixVK<double>& omega) const
{
	uint const vnum = value.vnum_;
	uint const knum = value.knum_;
	VectorK<double> const& alpha = *value.alpha_;
	MatrixKV<double> const& phi = *value.phi_;
	VectorV<uint> const& word_ct = *value.word_ct_;
	//VectorK<double> prev_gamma(value.knum_, 0);
	VectorK<double> sigma(value.knum_, 0);
	
	//while (!is_convergence(gamma, prev_gamma)){
	for (uint i = 0; i < max_local_iteration_; ++i){
		const auto exp_digamma = sig::map([](double g){ return std::exp(digamma(g)); }, gamma);

		for (uint v = 0; v<vnum; ++v){
			for (uint k = 0; k<knum; ++k){
				// update omega
				omega[v][k] = phi[k][v] * exp_digamma[k];
			}
			sig::normalize_dist(omega[v]);
			const uint wn = word_ct[v];
			sig::compound_assignment([wn](double& s, double p){ s += wn * p; }, sigma, omega[v]);
		}
		//prev_gamma = std::move(gamma);
		gamma = sig::plus(alpha, sigma);
	}
}


void MrLDA::init(bool resume)
{
	doc_word_ct_ = SIG_INIT_MATRIX(uint, D, V, 0);
	for (auto const& t : input_data_->tokens_){
		++doc_word_ct_[t.doc_id][t.word_id];
	}

	term_score_ = SIG_INIT_MATRIX(double, K, V, 0);

	auto base_pass = sig::modify_dirpass_tail(input_data_->working_directory_, true);

	if (resume){
		auto load_info = sig::read_line<std::string>(base_pass + resume_info_fname);
		if (sig::is_container_valid(load_info)){
			auto info = sig::fromJust(load_info);
			total_iter_ct_ = std::stoul(info[0]);
		}

		auto tmp_alpha = std::move(alpha_);
		auto load_alpha = sig::read_num<VectorK<double>>(base_pass + resume_alpha_fname, " ");
		if (sig::is_container_valid(load_alpha)){
			alpha_ = std::move(sig::fromJust(load_alpha));
			std::cout << "resume alpha" << std::endl;
		}
		else{
			alpha_ = std::move(tmp_alpha);
			std::cout << "resume alpha error : alpha is set as default" << std::endl;
		}
	}

	auto load_gamma = resume ? sig::read_num<MatrixDK<double>>(base_pass + resume_gamma_fname, " ") : nothing;
	if (sig::is_container_valid(load_gamma)){
		gamma_ = std::move(sig::fromJust(load_gamma));
		std::cout << "resume gamma" << std::endl;
	}
	else{
		gamma_ = MatrixDK<double>(D_, VectorK<double>(K_));
		for (DocumentId d = 0; d<D_; ++d){
			for (TopicId k=0; k<K_; ++k){
				gamma_[d][k] = alpha_[k] + rand_d_();
			}
			bool f = sig::normalize_dist(gamma_[d]);
		}
		if (resume) std::cout << "resume gamma error : gamma is set by random" << std::endl;
	}

	auto load_beta = resume ? sig::read_num<MatrixKV<double>>(base_pass + resume_phi_fname, " ") : nothing;
	if (sig::is_container_valid(load_beta)){
		phi_ = std::move(sig::fromJust(load_beta));
		std::cout << "resume phi" << std::endl;
	}
	else{
		phi_ = MatrixKV<double>(K_, VectorV<double>(V_));
		for (TopicId k = 0; k<K_; ++k){
			for (WordId v = 0; v<V_; ++v){
				phi_[k][v] = eta_[k][v] + rand_d_();
			}
			bool f = sig::normalize_dist(phi_[k]);
		}
		if (resume) std::cout << "resume phi error : phi is set by random" << std::endl;
	}
}

void MrLDA::saveResumeData() const
{
	std::cout << "save resume data... ";

	auto base_pass = input_data_->working_directory_;

	sig::save_num(alpha_, base_pass + resume_alpha_fname, " ");
	sig::save_num(gamma_, base_pass + resume_gamma_fname, " ");
	sig::save_num(phi_, base_pass + resume_phi_fname, " ");

	sig::clear_file(base_pass + resume_info_fname);
	sig::save_line(total_iter_ct_, base_pass + resume_info_fname, sig::WriteMode::append);

	std::cout << "completed" << std::endl;
}

double MrLDA::calcLiklihood(double term2, double term4) const
{
	return D_ * calcModule0(alpha_) + term2 + term3_ + term4;
}


void updateAlpha(VectorK<double>& alpha, VectorK<double> const& sufficient_statistics, const int D)
{
	const uint K = alpha.size();

	const auto calc_gradient = [D](double old_alpha, double suff_stat, double old_alpha_sum_digamma)
	{
		std::cout << "grad:" << D*old_alpha_sum_digamma << "," << -D*digamma(old_alpha) << "," << suff_stat << std::endl;
		return D * (old_alpha_sum_digamma - digamma(old_alpha)) + suff_stat;
	};

	const auto calc_hessian = [D](uint row, uint col, VectorK<double> const& old_alpha, double old_alpha_sum_trigamma)
	{
		return row == col ? D * (trigamma(old_alpha[col]) - old_alpha_sum_trigamma) : D * (-old_alpha_sum_trigamma);
	};

	auto regulate = [&](VectorK<double> const& delta, uint& decay) ->VectorK<double>
	{
		while (true){
			bool f = true;
			auto corr_delta = sig::multiplies(std::pow(0.8, decay), delta);

			std::cout << "delta:";
			for (uint i=0; i<K; ++i){
				std::cout << corr_delta[i] << ", ";
				if (alpha[i] < std::abs(corr_delta[i])){
					f = false;
					++decay;
					break;
				}
			}
			std::cout << std::endl;
			if (f) return corr_delta;
		}
	};

	uint iteration_count = 0;
	uint decay = 0;
	sig::ManageConvergence<VectorK<double>> convergence(alpha_convergence_threshold, sig::norm_L2);

	while (true) {
		const double alpha_sum = sig::sum(alpha);
		if (++iteration_count > max_alpha_update_iteration || convergence.update(alpha)) break;
		
		sig::vector_u<double> gradient(K);
		sig::matrix_u<double> hessian(K, K);
		auto alpha_sum_digamma = digamma(alpha_sum);
		auto alpha_sum_trigamma = trigamma(alpha_sum);

		for (uint i = 0; i < K; ++i){
			gradient(i) = calc_gradient(alpha[i], sufficient_statistics[i], alpha_sum_digamma);
			for (uint j = 0; j < K; ++j){
				hessian(i, j) = calc_hessian(i, j, alpha, alpha_sum_trigamma);
			}
		}

		auto delta = sig::from_vector_ublas(boost::numeric::ublas::prod(sig::fromJust(sig::invert_matrix(hessian)), gradient));
		auto corr_delta = regulate(delta, decay);

		sig::compound_assignment(
			[](double& a, double d){ a -= d; },
			alpha, corr_delta
		);

		std::cout << "alpha:";
		for (auto e : alpha) std::cout << e << ", ";
		std::cout << std::endl;
	}
	std::cout << "ct:" << iteration_count << std::endl;
}

/*
void updateAlpha(VectorK<double>& alpha, VectorK<double> const& sufficient_statistics, const int D)
{
	const uint K = alpha.size();
	uint iteration_count = 0;
	sig::ManageConvergence<VectorK<double>, sig::RelativeError, sig::Norm<1>> convergence(alpha_convergence_threshold, sig::norm_L1);

	VectorK<double> hessian(K);
	VectorK<double> gradient(K);
	VectorK<double> alpha_new(K);

	int decay = 0;
	while (true) {
		double sumG_H = 0;
		double sum1_H = 0;

		const double alpha_sum = sig::sum(alpha);
		if (++iteration_count > max_alpha_update_iteration || convergence.update(alpha)) break;

		for (int i = 0; i < K; i++) {
			gradient[i] =K * (digamma(alpha_sum) - digamma(alpha[i])) + sufficient_statistics[i];
			
			hessian[i] = -D * trigamma(alpha[i]);

			sumG_H += gradient[i] / hessian[i];
			sum1_H += 1 / hessian[i];
		}

		double z = D * trigamma(alpha_sum);
		double c = sumG_H / (1 / z + sum1_H);

		while (true) {
			bool singular = false;

			std::cout << "stepsize:";
			for (int i = 0; i < K; i++) {
				double step_size = std::pow(0.8, decay) * (gradient[i] - c) / hessian[i];
				std::cout << step_size << ", ";
				if (alpha[i] < std::abs(step_size)){
					// the current hessian matrix is singular
					singular = true;
					break;
				}
				alpha_new[i] = alpha[i] - step_size;
			}
			std::cout << std::endl;

			if (singular) {
				decay++;
				alpha_new = alpha;
			}
			else {
				break;
			}
		}
		alpha = alpha_new;
		std::cout << "alpha:" << alpha[0] << "," << alpha[1] << std::endl;
	}
	std::cout << "ct:" << iteration_count << std::endl;
}
*/

double MrLDA::iteration()
{
	auto updatePhi = [&](TopicId k, WordId v, double delta){
		phi_[k][v] = eta_[k][v] + delta;
	};
	
	double term2;
	VectorK<double> alpha_sufficient_statistics(K_, 0);
	
	mapreduce_->run<mapreduce::schedule_policy::cpu_parallel<mr_job>>(performance_result_);

	for (auto it = mapreduce_->begin_results(), end = mapreduce_->end_results(); it != end; ++it){
		if (std::get<0>(it->first) == ReduceKeyType::Lambda){
			updatePhi(std::get<1>(it->first), std::get<2>(it->first), it->second);
		}
		else if (std::get<0>(it->first) == ReduceKeyType::Alpha){
			alpha_sufficient_statistics[std::get<2>(it->first)] = it->second;
		}
		else{
			term2 = it->second;
		}
	}
	//updateAlpha(alpha_, alpha_sufficient_statistics, D_);

	for(auto& b : phi_) sig::normalize_dist(b);

	saveResumeData();

	return 0;//calcLiklihood(term2, sig::sum(lambda_, [&](VectorV<double> const& row){ return calcModule0(row); }));
}

void MrLDA::train(uint iteration_num, std::function<void(LDA const*)> callback)
{
	sig::ManageConvergenceSimple convergence(global_convergence_threshold);

	/*while (convergence.update(iteration())){
		std::cout << convergence.get_value() << std::endl;
	}*/

	for (uint i = 0; i < iteration_num; ++i, ++total_iter_ct_){
		iteration();
		calcTermScore(getPhi(), term_score_);
		save(Distribution::TOPIC, L"./test data");
		save(Distribution::TERM_SCORE, L"./test data");
		save(Distribution::DOCUMENT, L"./test data");
		callback(this);
		initMR();
	}
}


void MrLDA::save(Distribution target, FilepassString save_folder, bool detail) const
{
	save_folder = sig::modify_dirpass_tail(save_folder, true);

	switch (target){
	case Distribution::DOCUMENT:
		printTopic(getTheta(), input_data_->doc_names_, save_folder + SIG_STR_TO_FPSTR("document_mrlda"));
		break;
	case Distribution::TOPIC:
		printWord(getPhi(), std::vector<FilepassString>(), input_data_->words_, sig::maybe<uint>(20), save_folder + SIG_STR_TO_FPSTR("topic_mrlda"), detail);
		break;
	case Distribution::TERM_SCORE:
		printWord(getTermScore(), std::vector<FilepassString>(), input_data_->words_, sig::maybe<uint>(20), save_folder + SIG_STR_TO_FPSTR("term-score_mrlda"), detail);
		break;
	default:
		std::cout << "MrLDA::save error" << std::endl;
		getchar();
	}
}

auto MrLDA::getTheta() const->MatrixDK<double>
{
	MatrixDK<double> theta;

	for (DocumentId d = 0; d < D_; ++d){
		theta.push_back(getTheta(d));
	}
	return theta;
}

auto MrLDA::getTheta(DocumentId d_id) const->VectorK<double>
{
	double sum = sig::sum(gamma_[d_id]);
	// computed from the variational distribution
	return sig::map([sum](double g){ return g / sum; }, gamma_[d_id]);
}

auto MrLDA::getPhi() const->MatrixKV<double>
{
	MatrixKV<double> phi;

	for (TopicId k = 0; k < K_; ++k){
		phi.push_back(getPhi(k));
	}
	return phi;
}

auto MrLDA::getPhi(TopicId k_id) const->VectorV<double>
{
	/*double sum = sig::sum(lambda_[k_id]);
	// computed from the variational distribution
	return sig::map([sum](double l){ return l / sum; }, lambda_[k_id]);*/
	return phi_[k_id];
}

auto MrLDA::getWordOfTopic(Distribution target, uint return_word_num) const->VectorK< std::vector< std::tuple<std::wstring, double> > >
{
	VectorK< std::vector< std::tuple<std::wstring, double> > > result;

	for (TopicId k = 0; k < K_; ++k){
		result.push_back(getWordOfTopic(target, return_word_num, k));
	}
	return result;
}

auto MrLDA::getWordOfTopic(Distribution target, uint return_word_num, TopicId k_id) const->std::vector< std::tuple<std::wstring, double> >
{
	std::vector< std::tuple<std::wstring, double> > result;
	std::vector<double> df;

	if (target == Distribution::TOPIC) df = getPhi(k_id);
	else if (target == Distribution::TERM_SCORE) df = getTermScore(k_id);
	else{
		std::cout << "MrLDA::getWordOfTopic : Distributionが無効" << std::endl;
		return result;
	}
	return getTopWords(df, return_word_num, input_data_->words_);
}

auto MrLDA::getWordOfDocument(uint return_word_num) const->VectorD< std::vector< std::tuple<std::wstring, double> > >
{
	VectorD< std::vector< std::tuple<std::wstring, double> > > result;

	for (DocumentId d = 0; d<D_; ++d){
		result.push_back(getWordOfDocument(return_word_num, d));
	}
	return result;
}

auto MrLDA::getWordOfDocument(uint return_word_num, DocumentId d_id) const->std::vector< std::tuple<std::wstring, double> >
{
	std::vector< std::tuple<std::wstring, double> > result;
	auto top_wscore = getTermScoreOfDocument(d_id);

	for (uint i = 0; i<return_word_num; ++i){
		result.push_back(std::make_tuple(*input_data_->words_.getWord(std::get<0>(top_wscore[i])), std::get<1>(top_wscore[i])));
	}
	return std::move(result);
}

}