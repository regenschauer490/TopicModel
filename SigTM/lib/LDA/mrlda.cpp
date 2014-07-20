/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#include "mrlda.h"
#include "../helper/mapreduce_module.h"
#include "SigUtil/lib/calculation.hpp"
#include "SigUtil/lib/iteration.hpp"
#include "SigUtil/lib/distance/norm.hpp"
#include "SigUtil/lib/error_convergence.hpp"

namespace sigtm
{
const double MrLDA::local_convergence = 0.001;
const double MrLDA::global_convergence = 0.0001;

void MrLDA::MapTask::process(value_type const& value, VectorK<double>& gamma, MatrixVK<double>& phi) const
{
	uint const vnum = value.vnum_;
	uint const knum = value.knum_;
	VectorK<double> const& alpha = *value.alpha_;
	//MatrixKV<double> const& lambda = *value.lambda_;
	MatrixKV<double> const& beta = *value.beta_;
	VectorV<uint> const& word_ct = *value.word_ct_;
	//VectorK<double> prev_gamma(value.knum_, 0);
	VectorK<double> sigma(value.knum_, 0);
	
	//while (!is_convergence(gamma, prev_gamma)){
	for (uint i=0; i<100; ++i){
		const auto exp_digamma = sig::map([](double g){ return std::exp(digamma(g)); }, gamma);

		for (uint v = 0; v<vnum; ++v){
			for (uint k = 0; k<knum; ++k){
				// update phi
				phi[v][k] = beta[k][v] * exp_digamma[k];
			}
			sig::normalize_dist(phi[v]);
			const uint wn = word_ct[v];
			sig::compound_assignment([wn](double& s, double p){ s += wn * p; }, sigma, phi[v]);
		}
		//prev_gamma = std::move(gamma);
		gamma = sig::plus(alpha, sigma);
	}
}


void MrLDA::init()
{
	gamma_ = MatrixDK<double>(D_, VectorK<double>(K_));
	//lambda_ = MatrixKV<double>(K_, VectorV<double>(V_, 1.0 / K_));
	beta_ = MatrixKV<double>(K_, VectorV<double>(V_));
	doc_word_ct_ = MatrixDV<uint>(D_, VectorV<uint>(V_));
	term_score_ = MatrixKV<double>(K_, VectorV<double>(V_, 0));
	
	for (auto const& t : input_data_->tokens_){
		++doc_word_ct_[t.doc_id][t.word_id];
	}

	for (TopicId k = 0; k<K_; ++k){
		for (WordId v = 0; v<V_; ++v){
			beta_[k][v] = eta_[k][v] + rand_d_();
		}
		bool f = sig::normalize_dist(beta_[k]);
	}

	for (DocumentId d = 0; d<D_; ++d){
		for (TopicId k=0; k<K_; ++k){
			gamma_[d][k] = alpha_[k] + rand_d_();
		}
		bool f = sig::normalize_dist(gamma_[d]);
	}
}

double MrLDA::calcLiklihood(double term2, double term4) const
{
	return D_ * calcModule0(alpha_) + term2 + term3_ + term4;
}

double MrLDA::iteration()
{
/*	auto update_lambda = [&](TopicId k, WordId v, double delta){
			lambda_[k][v] = eta_[k][v] + delta;
	};*/
	auto update_beta = [&](TopicId k, WordId v, double delta){
		beta_[k][v] = eta_[k][v] + delta;
	};

	auto update_alpha = [](){};

	double term2;

	for (auto it = mapreduce_->begin_results(), end = mapreduce_->end_results(); it != end; ++it){
		if (std::get<0>(it->first) == ReduceKeyType::Lambda){
			//update_lambda(std::get<1>(it->first), std::get<2>(it->first), it->second);
			update_beta(std::get<1>(it->first), std::get<2>(it->first), it->second);
		}
		else if (std::get<0>(it->first) == ReduceKeyType::Alpha){
			update_alpha();
		}
		else{
			term2 = it->second;
		}
	}

	for(auto& b : beta_) sig::normalize_dist(b);

	return 0;//calcLiklihood(term2, sig::sum(lambda_, [&](VectorV<double> const& row){ return calcModule0(row); }));
}

void MrLDA::learn(uint iteration_num)
{
	sig::ManageConvergenceSimple convergence(global_convergence);
	//mapreduce_->run<mapreduce::schedule_policy::sequential<mr_job>>(performance_result_);
	
	/*while (convergence.update(iteration())){
		std::cout << convergence.get_value() << std::endl;
	}*/

	for (uint i = 0; i < iteration_num; ++i){
		mapreduce_->run<mapreduce::schedule_policy::cpu_parallel<mr_job>>(performance_result_);
		iteration();
		calcTermScore(getWordDistribution(), term_score_);
		save(Distribution::TOPIC, L"./test data");
		save(Distribution::TERM_SCORE, L"./test data");
		save(Distribution::DOCUMENT, L"./test data");
		initMR();
	}
}


void MrLDA::save(Distribution target, FilepassString save_folder, bool detail) const
{
	save_folder = sig::impl::modify_dirpass_tail(save_folder, true);

	switch (target){
	case Distribution::DOCUMENT:
		printTopic(getTopicDistribution(), save_folder + SIG_STR_TO_FPSTR("document_mrlda"));
		break;
	case Distribution::TOPIC:
		printWord(getWordDistribution(), input_data_->words_, sig::maybe<uint>(20), save_folder + SIG_STR_TO_FPSTR("topic_mrlda"), detail);
		break;
	case Distribution::TERM_SCORE:
		printWord(getTermScoreOfTopic(), input_data_->words_, sig::maybe<uint>(20), save_folder + SIG_STR_TO_FPSTR("term-score_mrlda"), detail);
		break;
	default:
		printf("\nforget: LDA_Gibbs::print\n");
		getchar();
	}
}

auto MrLDA::getTopicDistribution() const->MatrixDK<double>
{
	MatrixDK<double> theta;

	for (DocumentId d = 0; d < D_; ++d){
		theta.push_back(getTopicDistribution(d));
	}
	return theta;
}

auto MrLDA::getTopicDistribution(DocumentId d_id) const->VectorK<double>
{
	double sum = sig::sum(gamma_[d_id]);
	// computed from the variational distribution
	return sig::map([sum](double g){ return g / sum; }, gamma_[d_id]);
}

auto MrLDA::getWordDistribution() const->MatrixKV<double>
{
	MatrixKV<double> beta;

	for (TopicId k = 0; k < K_; ++k){
		beta.push_back(getWordDistribution(k));
	}
	return beta;
}

auto MrLDA::getWordDistribution(TopicId k_id) const->VectorV<double>
{
	/*double sum = sig::sum(lambda_[k_id]);
	// computed from the variational distribution
	return sig::map([sum](double l){ return l / sum; }, lambda_[k_id]);*/
	return beta_[k_id];
}

auto MrLDA::getTermScoreOfTopic() const->MatrixKV<double>
{
	return term_score_;
}

auto MrLDA::getTermScoreOfTopic(TopicId t_id) const->VectorV<double>
{
	return term_score_[t_id];
}

auto MrLDA::getTermScoreOfDocument(DocumentId d_id) const->std::vector< std::tuple<WordId, double> >
{
	const auto theta = getTopicDistribution(d_id);
	const auto tscore = getTermScoreOfTopic();

	VectorV<double> tmp(V_, 0.0);
	TopicId t = 0;

	for (auto d1 = theta.begin(), d1end = theta.end(); d1 != d1end; ++d1, ++t){
		WordId w = 0;
		for (auto d2 = tscore[t].begin(), d2end = tscore[t].end(); d2 != d2end; ++d2, ++w){
			tmp[w] += ((*d1) * (*d2));
		}
	}

	auto sorted = sig::sort_with_index(tmp); //std::tuple<std::vector<double>, std::vector<uint>>
	return sig::zipWith([](WordId w, double d){ return std::make_tuple(w, d); }, std::get<1>(sorted), std::get<0>(sorted)); //sig::zip(std::get<1>(sorted), std::get<0>(sorted));
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

	if (target == Distribution::TOPIC) df = getWordDistribution(k_id);
	else if (target == Distribution::TERM_SCORE) df = getTermScoreOfTopic(k_id);
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