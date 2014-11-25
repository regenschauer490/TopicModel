/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_CTR_HPP
#define SIGTM_CTR_HPP

#define _SCL_SECURE_NO_WARNINGS

#include "../helper/data_format.hpp"
#include "../helper/rating_matrix.hpp"
#include "lda_common_module.hpp"
#include "SigUtil/lib/calculation/ublas.hpp"
//#include <boost/numeric/ublas/matrix_sparse.hpp>

namespace sigtm
{

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
		alpha_smooth_ = 1.0;
		beta_smooth_ =default_beta;
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
	using RatingIter = SparseBooleanMatrix::const_iterator;
	using RatingContainer = SparseBooleanMatrix::const_rating_range;
	
	const CtrHyperparameter hparam_;
	const DocumentSetPtr input_data_;
	const TokenList& tokens_;

	const VectorI<std::vector<TokenId>> item_tokens_;	// tokens in each item(document)

	const RatingContainer user_ratings_;
	const RatingContainer item_ratings_;

	const uint T_;		// number of tokens
	const uint K_;		// number of topics(factor)
	const uint V_;		// number of words	
	const uint U_;		// number of users
	const uint I_;		// number of items

	MatrixKV_<double> beta_;	// word distribution of topic
	MatrixIK_<double> theta_;
	MatrixUK_<double> user_factor_;
	MatrixIK_<double> item_factor_;

	double likelihood_;
	const double conv_epsilon_ = 1e-4;

	// temporary
	VectorK_<double> gamma_;
	MatrixKV_<double> log_beta_;
	MatrixKV_<double> word_ss_;
	MatrixTK_<double> phi_;

private:
	void init();

	void saveTmp() const;
	void save() const;
	void load();

	double docInference(ItemId id, bool update_word_ss);

	void updateU();
	void updateV();
	void updateBeta();

private:
	CTR(uint topic_num, CtrHyperparameter hparam, DocumentSetPtr docs, RatingContainer user_ratings, RatingContainer item_ratings)
	: hparam_(hparam), input_data_(docs), tokens_(docs->tokens_), item_tokens_(docs->getDevidedDocument()),
		user_ratings_(std::move(user_ratings)), item_ratings_(std::move(item_ratings)), T_(docs->getTokenNum()), K_(topic_num), V_(docs->getWordNum()),
		U_(user_ratings_.size()), I_(item_ratings_.size()), beta_(K_, V_), theta_(I_, K_), user_factor_(U_, K_), item_factor_(I_, K_), likelihood_(-std::exp(50)),
		gamma_(K_), log_beta_(K_, V_), word_ss_(K_, V_), phi_(T_, K_)
	{
		init();
	}
	
public:	
	static auto makeInstance(uint topic_num, CtrHyperparameter hparam, DocumentSetPtr docs, BooleanMatrixPtr ratings) ->std::shared_ptr<CTR>
	{
		return std::shared_ptr<CTR>(new CTR(topic_num, hparam, docs, ratings->getUsers(), ratings->getItems()));
	}

	void train(uint max_iter, uint min_iter, uint save_lag);
};

using CTRPtr = std::shared_ptr<CTR>;
}	// sigtm


#include "../helper/metrics.hpp"
#include "../helper/cross_validation.hpp"

namespace sigtm
{
template <>
struct Precision<CTR> : public PrecisionBase
{
	using const_iterator_range =SparseRatingMatrixBase_::const_iterator_range;
	
	// precision for user (item recommendation)
	double operator()(CTRPtr model, const_iterator_range test_set) const
	{
	
	
		std::unordered_set<UserId> check;
		
		for(auto const& e : ){
			UserId uid = e->user_id_;
			if(check.count(uid)) continue;
			
			check.emplace(uid);
			auto est = model->estimate(uid);
			
		}
		return impl(model->estimate(), test_set, true, [&](RatingPtr const& r1, RatingPtr const& r2){ return r1->item_id_ < r2->item_id_; });
	}
};

template <>
class CrossValidation<CTR> : public CrossValidationBase
{
	uint topic_num_;
	CtrHyperparameter hparam_;
	DocumentSetPtr docs_;
	
public:
	CrossValidation(uint n, uint topic_num, CtrHyperparameter hparam, DocumentSetPtr docs, BooleanMatrixPtr ratings)
		: topic_num_(topic_num), hparam_(hparam), docs_(docs)
	{
		rating_chunks_ = devide_rating_matrix(n, ratings);
	}
	
	// F: R evaluation_func(CTRPtr model, const_iterator test_begin, const_iterator test_end)
	template <class F>
	auto run(F&& evaluation_func, uint max_iter, uint min_iter, uint save_lag)
	{
		using R = decltype(sig::impl::eval(
			std::forward<F>(evaluation_func),
			std::declval<CTR>(), 
			std::declval<const_iterator>(),
			std::declval<const_iterator>()
		));
	
		auto validation = [&](uint test_index)
		{
			auto model = CTR::makeInstance(topic_num_, hparam_, docs_, BooleanMatrix::makeInstance(join_chunc(rating_chunks_, test_index)));
			
			model.tarin(max_iter, min_iter, save_lag);
			
			return std::forward<F>(evaluation_func)(model, rating_chunks_[i]);
		};		
		
		std::vector<R> result;
				
		for(uint i = 1; i < n; ++i){
			result.push_back(validation(i));
		}
		
		return result;
	}
};


}
#endif