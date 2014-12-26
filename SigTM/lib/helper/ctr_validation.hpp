/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_CTR_VALIDATION_HPP
#define SIGTM_CTR_VALIDATION_HPP

#include "../model/ctr.h"
#include "../helper/metrics.hpp"
#include "../helper/cross_validation.hpp"
#include "SigUtil/lib/functional/filter.hpp"
//#include <boost/numeric/ublas/matrix_sparse.hpp>

namespace sigtm
{

template <class Derived>
struct CTR_PR_IMPL
{
	using RatingContainer_ = RatingContainer<CTR::RatingValueType>;

	CTR_PR_IMPL(sig::Maybe<uint> top_n, sig::Maybe<double> threshold) : top_n_(top_n), threshold_(threshold){}
	virtual ~CTR_PR_IMPL() = default;

	auto operator()(CTRPtr model, std::vector<RatingContainer_>& test_set, bool is_user_test) const->double
	{
		std::vector<double> result;

		if (is_user_test){
			for (UserId id = 0, size = test_set.size(); id < size; ++id){
				auto est = model->recommend(id, true, top_n_, threshold_);

				auto val = static_cast<const Derived*>(this)->impl(
					sig::map([](CTR::EstValueType const& e) { return e.first; }, est),
					sig::map([](CTR::RatingPtr_ const& e) { return e->item_id_; }, test_set[id]),// sig::filter([](RatingContainer_::value_type const& r) { return !r->is_duplicate_; }, test_set[id])),
					false,
					std::less<uint>()
				);
				if(val){
					//std::cout << "user:" << id << ", val:" << *val << ", #ans:" << test_set[id] .size() << ", #est:" << est.size() << std::endl;
					result.push_back(*val);
				}
			}
		}
		else{
			/*for (ItemId id = 0, size = test_set.size(); id < size; ++id){
				auto val = static_cast<const Derived*>(this)->impl(
					sig::map([](CTR::EstValueType const& e){ return e.first; }, model->recommend(id, false, top_n_, threshold_)),
					sig::map([](CTR::RatingPtr_ const& e){ return e->user_id_; }, test_set[id]),
					false,
					std::less<uint>()
				);
				if(val) result.push_back(*val);
			}*/
		}
		return sig::average(result);
	}

private:
	sig::Maybe<uint> top_n_;
	sig::Maybe<double> threshold_;
};

// precision for user (item recommendation)
template <>
struct Precision<CTR> : public PrecisionBase, public CTR_PR_IMPL<Precision<CTR>>
{
	Precision(sig::Maybe<uint> top_n, sig::Maybe<double> threshold) : CTR_PR_IMPL(top_n, threshold){}
};

template <>
struct Recall<CTR> : public RecallBase, public CTR_PR_IMPL<Recall<CTR>>
{
	Recall(sig::Maybe<uint> top_n, sig::Maybe<double> threshold) : CTR_PR_IMPL(top_n, threshold){}
};

template <>
struct F_Measure<CTR> : public F_MeasureBase, private CTR_PR_IMPL<Precision<CTR>>, private CTR_PR_IMPL<Recall<CTR>>
{
	F_Measure(sig::Maybe<uint> top_n, sig::Maybe<double> threshold) : CTR_PR_IMPL<Precision<CTR>>(top_n, threshold), CTR_PR_IMPL<Recall<CTR>>(top_n, threshold){}

	double operator()(CTRPtr model, std::vector<RatingContainer<CTR::RatingValueType>>& test_set, bool is_user_test) const
	{
		return impl(
			CTR_PR_IMPL<Precision<CTR>>::operator()(model, test_set, is_user_test),
			CTR_PR_IMPL<Recall<CTR>>::operator()(model, test_set, is_user_test)
		);
	}
};


template <>
class CrossValidation<CTR> : public CrossValidationBase<CTR::RatingValueType>
{
	std::vector<CTRPtr> models_;
	bool is_user_test_;
	uint topic_num_;
	CTRHyperParamPtr hparam_;
	DocumentSetPtr docs_;
	
public:
	CrossValidation(uint n, bool is_user_test, uint topic_num, CTRHyperParamPtr hparam, DocumentSetPtr docs, RatingMatrixPtr<CTR::RatingValueType> ratings, uint max_iter, uint min_iter, uint save_lag)
		: CrossValidationBase(n), models_(0), is_user_test_(is_user_test), topic_num_(topic_num), hparam_(hparam), docs_(docs)
	{
		//rating_chunks_ = devide_random(n, *ratings, is_user_test_);
		rating_chunks_ = devide_adjusted_random(n, *ratings, is_user_test_);

		auto train_model = [=](uint i){
			auto model = CTR::makeInstance(topic_num_, hparam_, docs_, SparseBooleanMatrix::makeInstance(join_chunc(rating_chunks_, i)), i);
			model->train(max_iter, min_iter, save_lag);

			return model;
		};

		models_ = run_parallel<CTRPtr>(train_model, n, sig::seqn(0u, 1u, div_num_).begin());
	}
	
	// F: R evaluation_func(CTRPtr model, std::vector<RatingContainer_>& test_set, bool is_user_test, double threshold)
	template <class F>
	auto run(F&& evaluation_func)
	{
		using R = decltype(sig::impl::eval(
			std::forward<F>(evaluation_func),
			std::declval<CTRPtr>(), 
			rating_chunks_[0],
			is_user_test_
		));
	
		auto validation = [&](uint test_index)
		{
			return std::forward<F>(evaluation_func)(models_[test_index], rating_chunks_[test_index], is_user_test_);
		};		
		
		std::vector<R> result;

		return run_parallel<R>(validation, div_num_, sig::seqn(0u, 1u, div_num_).begin());
	}


	void debug_set_u(std::vector<std::vector<double>> v){ for(auto& m : models_) m->debug_set_u(v); }
	void debug_set_v(std::vector<std::vector<double>> v){ for(auto& m : models_) m->debug_set_v(v); }
};


}
#endif