
#ifndef SIGTM_CTR_HPP
#define SIGTM_CTR_HPP

#include "../sigtm.hpp"
#include "../helper/data_format.hpp"
#include "lda_common_module.hpp"
#include "SigUtil/lib/calculation/ublas.hpp"

namespace sigtm
{

using ItemId = uint;
template<class T> using VectorI = VectorD<T>;			// item
template<class T> using VectorU = std::vector<T>;		// user
template<class T> using VectorK_ = sig::vector_u<T>;	// topic
template<class T> using MatrixIK = VectorI<VectorK<T>>;	// item - topic(factor)

template<class T> using MatrixIK_ = sig::matrix_u<T>;	// item - topic(factor)
template<class T> using MatrixUK_ = sig::matrix_u<T>;	// user - topic(factor)
template<class T> using MatrixKK_ = sig::matrix_u<T>;
template<class T> using MatrixKV_ = sig::matrix_u<T>;
template<class T> using MatrixTK_ = sig::matrix_u<T>;

struct CtrHyperparameter
{
	double a_;				// positive item weight
	double b_;				// negative item weight(b < a)
	double lambda_u_;
	double lambda_v_;
	double learning_rate_;	// stochastic version for large datasets. Stochastic learning will be called when > 0
	double alpha_smooth_;
	double beta_smooth_;
	bool theta_opt_;
	bool lda_regression_;

	CtrHyperparameter(bool optimize_theta, bool run_lda_regression)
	{
		a_ = 1;
		b_ = 0.01;
		lambda_u_ = 0.01;
		lambda_v_ = 100;
		learning_rate_ = -1;
		alpha_smooth_ = alpha_smooth_;
		beta_smooth_ = beta_smooth_;
		theta_opt_ = optimize_theta;
		lda_regression_ = run_lda_regression;
	}
	/*
	void set(double a, double b, double lu, double lv, double lr, double as, int to, int lda_r)
	{
		a_ = a;
		b_ = b;
		lambda_u_ = lu;
		lambda_v_ = lv;
		learning_rate_ = lr;
		alpha_smooth_ = as;
		theta_opt_ = to;
		lda_regression_ = lda_r;
	}*/
};


class CTR
{
	const CtrHyperparameter hparam_;
	const DocumentSetPtr input_data_;
	const TokenList& tokens_;

	const VectorI<std::vector<TokenId>> item_token_;	// tokens in each item(document)

	const VectorU<std::vector<ItemId>>& user_rating_;
	const VectorI<std::vector<UserId>>& item_rating_;

	const uint T_;		// number of tokens
	const uint K_;		// number of topics(factor)
	const uint V_;		// number of words	
	const uint U_;		// number of users
	const uint I_;		// number of items

	MatrixKV<double> beta_;	// word distribution of topic
	MatrixIK<double> theta_;
	MatrixUK_<double> user_factor_;
	MatrixIK_<double> item_factor_;

	double likelihood_;
	const double conv_epsilon_ = 1e-4;

	// temporary
	VectorK_<double> gamma_;
	MatrixKV_<double> log_beta_;
	MatrixKV<double> word_ss_;
	MatrixTK<double> phi_;

private:
	void init();

	void saveTmp() const;
	void save() const;
	void load();

	double docInference(ItemId id, bool update_word_ss);

	void updateU();
	void updateV();
	void updateBeta();

	void update();

public:
	CTR(uint topic_num, CtrHyperparameter hparam, DocumentSetPtr docs, VectorU<std::vector<ItemId>> const& user_ratings, VectorI<std::vector<UserId>> const& item_ratings) :
		hparam_(hparam), input_data_(docs), tokens_(docs->tokens_), item_token_(docs->getDevidedDocument()),
		user_rating_(user_ratings), item_rating_(item_ratings), T_(docs->getTokenNum()), K_(topic_num), V_(docs->getWordNum()),
		U_(user_ratings.size()), I_(item_ratings.size()), likelihood_(-std::exp(50))
	{
		init();
	}
};

}

#endif