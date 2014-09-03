/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_TWITTER_LDA_H
#define SIGTM_TWITTER_LDA_H

#include "lda_interface.hpp"
#include "SigUtil/lib/array.hpp"

#include "../helper/input.h"
#if USE_SIGNLP
#include "../helper/input_text.h"
#endif


namespace sigtm
{

template<class T> using VectorB = sig::array<T, 2>;	// for bernoulli parameter
template<class T> using VectorU = std::vector<T>;	// all users
template<class T> using MatrixUD = VectorU<VectorD<T>>;	// user - tweet
template<class T> using MatrixDT = VectorD<VectorT<T>>;	// tweet - token

class TwitterLDA;
using TwitterLDAPtr = std::shared_ptr<TwitterLDA>;

class TwitterLDA
{
	InputDataPtr input_data_;
	TokenList const& tokens_;	// ※ソート操作による元データ変更の可能性あり

	const uint U_;				// number of users
	const uint K_;				// number of topics
	const uint V_;				// number of words
	const VectorU<UserId> D_;	// number of tweets in each user
	const MatrixUD<Id> T_;		// number of tokens in each tweet

	VectorK<double> alpha_;			// dirichlet hyper parameter of theta
	VectorV<double> beta_;			// dirichlet hyper parameter of phi
	VectorB<double> gamma_;			// dirichlet hyper parameter of pi

	VectorD<uint> z_;				// topic assigned to each token
	MatrixDT<bool> y_;				// choice between background words and topic words

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
	TwitterLDA() = delete;
	TwitterLDA(TwitterLDA const&) = delete;

	TwitterLDA(bool resume, uint topic_num, InputDataPtr input_data, maybe<VectorK<double>> alpha, maybe<VectorV<double>> beta, maybe<VectorB<double>> gamma) :
		input_data_(input_data), tokens_(input_data->tokens_), U_(input_data->getDocNum()), K_(topic_num), V_(input_data->getWordNum()),
		alpha_(alpha ? sig::fromJust(alpha) : SIG_INIT_VECTOR(double, K, default_alpha_base / K_)), beta_(beta ? sig::fromJust(beta) : SIG_INIT_VECTOR(double, V, default_beta)), gamma_(gamma ? sig::fromJust(gamma) : VectorB<double>{0.5, 0.5}),
		word_ct_(SIG_INIT_MATRIX(uint, V, K, 0)), doc_ct_(SIG_INIT_MATRIX(uint, D, K, 0)), topic_ct_(SIG_INIT_VECTOR(uint, K, 0)),
		alpha_sum_(0), beta_sum_(0), tmp_p_(SIG_INIT_VECTOR(double, K, 0)), z_(tokens_.size(), 0), term_score_(SIG_INIT_MATRIX(double, K, V, 0)), total_iter_ct_(0),
		sampling_(SamplingMethod()), rand_ui_(0, K_ - 1, FixedRandom), rand_d_(0.0, 1.0, FixedRandom)
	{
		init(resume);
	}

	void init(bool resume);
	void updateY(Token const& t);
	void saveResumeData() const;

	
public:
	/* InputDataで作成した入力データを元にコンストラクト */
	// デフォルト設定で使用する場合
	static TwitterLDAPtr makeInstance(bool resume, uint topic_num, InputDataPtr input_data){
		return DocumentType::Tweet == input_data->doc_type_
			? TwitterLDAPtr(new TwitterLDA(resume, topic_num, input_data, nothing, nothing))
			: nullptr;
	}
	// alpha, beta をsymmetricに設定する場合
	template <class SamplingMethod = CollapsedGibbsSampling>
	static LDAPtr makeInstance(bool resume, uint topic_num, InputDataPtr input_data, double alpha, maybe<double> beta = nothing){
		return LDAPtr(new LDA_Gibbs(SamplingMethod(), resume, topic_num, input_data, VectorK<double>(topic_num, alpha), beta ? sig::Just<VectorV<double>>(VectorV<double>(input_data->getWordNum(), sig::fromJust(beta))) : nothing));
	}
	// alpha, beta を多次元で設定する場合
	template <class SamplingMethod = CollapsedGibbsSampling>
	static LDAPtr makeInstance(bool resume, uint topic_num, InputDataPtr input_data, VectorK<double> alpha, maybe<VectorV<double>> beta = nothing){
		return LDAPtr(new LDA_Gibbs(SamplingMethod(), resume, topic_num, input_data, alpha, beta));
	}

};

}
#endif