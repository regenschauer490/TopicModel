/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_TWITTER_LDA_H
#define SIGTM_TWITTER_LDA_H

#include "lda_interface.hpp"
#include "../helper/input.h"

#if USE_SIGNLP
#include "../helper/input_text.h"
#endif


namespace sigtm
{

template<class T> using VectorU = std::vector<T>;	// all users
template<class T> using MatrixUD = VectorU<VectorD<T>>;	// user - tweet

class TwitterLDA
{
	InputDataPtr input_data_;
	TokenList const& tokens_;

	const uint U_;				// number of users
	const uint K_;				// number of topics
	const uint V_;				// number of words
	const VectorU<UserId> D_;	// number of tweets in each user
	const MatrixUD<Id> T_;		// number of tokens in each tweet

	VectorK<double> alpha_;			// dirichlet hyper parameter of theta
	VectorV<double> beta_;			// dirichlet hyper parameter of phi
	VectorT<uint> z_;				// topic assigned to each tokens temporary

	MatrixVK<uint> word_ct_;		// topic count of each word
	MatrixDK<uint> doc_ct_;			// topic count of each document
	VectorK<uint> topic_ct_;		// topic count of all token

	double alpha_sum_;
	double beta_sum_;
	VectorK<double> tmp_p_;
	MatrixKV<double> term_score_;	// word score of emphasizing each topic
	uint total_iter_ct_;

	const std::function<double(LDA_Gibbs const* obj, Token const& t, uint k)> sampling_;
	sig::SimpleRandom<uint> rand_ui_;
	sig::SimpleRandom<double> rand_d_;

private:

	
public:
};

}
#endif