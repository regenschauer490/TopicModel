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
		a_ = 2;
		b_ = 0.01;
		lambda_u_ = 0.01;
		lambda_v_ = 100;
		learning_rate_ = -1;
		alpha_smooth_ = 0.0;
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

//template <class RatingValueType>
class CTR
{
public:
	using RatingValueType = int;
	using RatingPtr_ = RatingPtr<RatingValueType>;
	using EstValueType = double;
	using EstRatingPtr_ = RatingPtr<EstValueType>;

private:
	using RatingIter = SparseBooleanMatrix::const_iterator;
	using RatingContainer = SparseBooleanMatrix::const_rating_range;
	
	const CtrHyperparameter hparam_;
	const DocumentSetPtr input_data_;
	const RatingMatrixPtr<RatingValueType> ratings_;
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

	void printUFactor() const;
	void printIFactor() const;

	void saveTmp() const;
	void save() const;
	void load();

	double docInference(ItemId id, bool update_word_ss);

	void updateU();
	void updateV();
	void updateBeta();
	
private:
	CTR(uint topic_num, CtrHyperparameter hparam, DocumentSetPtr docs, RatingMatrixPtr<RatingValueType> ratings)
	: hparam_(hparam), input_data_(docs), ratings_(ratings), tokens_(docs->tokens_), item_tokens_(docs->getDevidedDocument()),
		user_ratings_(ratings->getUsers()), item_ratings_(ratings->getItems()), T_(docs->getTokenNum()), K_(topic_num), V_(docs->getWordNum()),
		U_(ratings->userSize()), I_(ratings->itemSize()), beta_(K_, V_), theta_(I_, K_), user_factor_(U_, K_), item_factor_(I_, K_), likelihood_(-std::exp(50)),
		gamma_(K_), log_beta_(K_, V_), word_ss_(K_, V_), phi_(T_, K_)
	{
		init();
	}
	
public:	
	static auto makeInstance(uint topic_num, CtrHyperparameter hparam, DocumentSetPtr docs, RatingMatrixPtr<RatingValueType> ratings) ->std::shared_ptr<CTR>
	{
		return std::shared_ptr<CTR>(new CTR(topic_num, hparam, docs, ratings));
	}

	void train(uint max_iter, uint min_iter, uint save_lag);

	// return recommended item(for user) or user(for item) ids (descending by estimated rating value)
	auto recommend(Id id, bool for_user, double threshold) const->std::vector<Id>;
	auto recommend_detail(Id id, bool for_user, double threshold) const->std::vector<EstRatingPtr_>;

	double estimate(UserId u_id, ItemId i_id) const;
};

using CTRPtr = std::shared_ptr<CTR>;
}	// sigtm


#include "../helper/metrics.hpp"
#include "../helper/cross_validation.hpp"

namespace sigtm
{

template <class Derived>
struct CTR_PR_IMPL
{
	using RatingPtr_ = CTR::RatingPtr_;
	using RatingContainer_ = RatingContainer<CTR::RatingValueType>;

	auto operator()(CTRPtr model, std::vector<RatingContainer_>& test_set, bool is_user_test, double threshold) const->std::vector<double>
	{
		std::vector<double> result;

		if (is_user_test){
			for (UserId id = 0, size = test_set.size(); id < size; ++id){
				auto val = static_cast<const Derived*>(this)->impl(
					model->recommend(id, true, threshold),
					sig::map([](RatingPtr_ const& e){ return e->item_id_; }, test_set[id]),
					false,
					std::less<double>()
				);
				if(val) result.push_back(*val);
			}
		}
		else{
			for (ItemId id = 0, size = test_set.size(); id < size; ++id){
				auto val = static_cast<const Derived*>(this)->impl(
					model->recommend(id, false, threshold),
					sig::map([](RatingPtr_ const& e){ return e->user_id_; }, test_set[id]),
					false,
					std::less<double>()
				);
				if(val) result.push_back(*val);
			}
		}

		return result;
	}
};

// precision for user (item recommendation)
template <>
struct Precision<CTR> : public PrecisionBase, public CTR_PR_IMPL<Precision<CTR>>
{};

template <>
struct Recall<CTR> : public RecallBase, public CTR_PR_IMPL<Recall<CTR>>
{};


template <>
class CrossValidation<CTR> : public CrossValidationBase<CTR::RatingValueType>
{
	std::vector<CTRPtr> models_;
	bool is_user_test_;
	uint topic_num_;
	CtrHyperparameter hparam_;
	DocumentSetPtr docs_;
	
public:
	CrossValidation(uint n, bool is_user_test, uint topic_num, CtrHyperparameter hparam, DocumentSetPtr docs, RatingMatrixPtr<CTR::RatingValueType> ratings, uint max_iter, uint min_iter, uint save_lag)
		: CrossValidationBase(n), models_(n), is_user_test_(is_user_test), topic_num_(topic_num), hparam_(hparam), docs_(docs)
	{
		rating_chunks_ = random_devide(n, *ratings, is_user_test_);

		auto seq = sig::seqn(0u, 1u, div_num_);

		for (auto e : seq){
			std::cout << e << std::endl;
		}

		auto train_model = [&](uint i)->CTRPtr
		{
			auto model = CTR::makeInstance(topic_num_, hparam_, docs_, SparseBooleanMatrix::makeInstance(join_chunc(rating_chunks_, i)));
			model->train(max_iter, min_iter, save_lag);

			return model;
		};

		models_ = run_parallel<CTRPtr>(train_model, div_num_, seq.begin());
	}
	
	// F: R evaluation_func(CTRPtr model, const_iterator test_begin, const_iterator test_end)
	template <class F>
	auto run(F&& evaluation_func, double threshold)
	{
		using R = decltype(sig::impl::eval(
			std::forward<F>(evaluation_func),
			std::declval<CTRPtr>(), 
			rating_chunks_[0],
			is_user_test_,
			threshold
		));
	
		auto validation = [&](uint test_index)
		{
			return std::forward<F>(evaluation_func)(models_[test_index], rating_chunks_[test_index], is_user_test_, threshold);
		};		
		
		std::vector<R> result;
				
		//return run_parallel<R>(validation, div_num_, sig::seqn(0, 1, div_num_).begin());

		for(uint i = 1; i < div_num_; ++i){
			sig::TimeWatch tw;
			result.push_back(validation(i));
			tw.save();
			sig::save_line(std::to_string(tw.get_total_time<std::chrono::seconds>()), L"./time.txt", sig::WriteMode::append);
		}
		
		return result;
	}
};


}
#endif