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

template <class C>
void makeRandomDistribution(uint elem_num) ->C
{
	C result;
	for (uint i = 0; i < elem_num; ++i){
		sig::container_traits<C>::add_element(result, rand_d_());
	}
	sig::normalize_dist(result);

	return result;
}

void LDA_CVB::init(bool resume)
{
	int i = -1;
	lambda_ = SIG_INIT_MATRIX(double, V, K);
	gamma_ = SIG_INIT_MATRIX(double, D, K);
	topic_sum_ = SIG_INIT_VECTOR(double, K);

	for (auto const& t : tokens_){
		auto omega_rand = makeRandomDistribution<VectorK<double>>(K_);
		sig::compound_assignment(sig::assign_plus<double>(), lambda_[t.word_id], omega_rand);
		sig::compound_assignment(sig::assign_plus<double>(), gamma_[t.doc_id], omega_rand);
		sig::compound_assignment(sig::assign_plus<double>(), topic_sum_, omega_rand);
	}
}

void LDA_CVB::update(Token const& t)
{
	//トピック比率の更新
	const auto updateTopic = [&](Token const& t)
	{
		for (TopicId k = 0; k < K_; ++k){
			WordId v = t.word_id;
			omega_[t.self_id][k] = (gamma_[t.doc_id][k] + alpha_[k]) * (lambda_[v][k] + beta_[k][v]) / (topic_sum_[k] + V_ * beta_[k][v]);
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


auto LDA_CVB::getTheta(DocumentId d_id) const->VectorK<double>
{
	double sum = sig::sum(gamma_[d_id]);
	return sig::map([sum](double v){ return v / sum; }, gamma_[d_id]);
}

auto LDA_CVB::getPhi(TopicId k_id) const->VectorV<double>
{
	double sum = sig::sum_col(lambda_, k_id);
	// computed from the variational distribution
	return sig::map([&](uint i){ return lambda_[i][k_id] / sum; }, sig::seq(0, 1, V_));
}

}