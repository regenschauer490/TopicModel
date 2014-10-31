
#ifndef SIGTM_CTR_HPP
#define SIGTM_CTR_HPP

#include "../sigtm.hpp"
#include "../helper/data_format.hpp"
#include "lda_common_module.hpp"
#include "SigUtil/lib/calculation/ublas.hpp"

namespace sigtm
{

using ItemId = uint;
template<class T> using VectorI = std::vector<T>;	// item
template<class T> using VectorU = std::vector<T>;	// user
template<class T> using VectorK_ = sig::vector_u<T>;	//  
template<class T> using MatrixIK = sig::matrix_u<T>;	// item - topic(factor)
template<class T> using MatrixUK = sig::matrix_u<T>;	// user - topic(factor)
template<class T> using MatrixKK_ = sig::matrix_u<T>;
template<class T> using MatrixKV_ = sig::matrix_u<T>;
template<class T> using MatrixTK_ = sig::matrix_u<T>;

struct CtrHyperparameter
{
	double a_;				// positive item weight, default 1
	double b_;				// negative item weight, default 0.01 (b < a)
	double lambda_u_;
	double lambda_v_;
	double learning_rate_;	// stochastic version for large datasets, default -1. Stochastic learning will be called when > 0
	double alpha_smooth_;
	bool theta_opt_;
	bool lda_regression_;

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
	}
};


class CTR
{
	const CtrHyperparameter& hparam_;
	const DocumentSetPtr input_data_;
	const VectorI<std::vector<TokenId>> item_token_;

	const VectorU<std::vector<ItemId>> user_rating_;
	const VectorI<std::vector<UserId>> item_rating_;

	const uint T_;		// number of tokens
	const uint K_;		// number of topics(factor)
	const uint V_;		// number of words	
	const uint U_;		// number of users
	const uint I_;		// number of items

	MatrixUK<double> user_factor_;
	MatrixIK<double> item_factor_;
	MatrixIK<double> theta_;

	double likelihood_;

private:

	void updateU();
	void updateV();
	void updateBeta();

	void update();
};

}
#endif