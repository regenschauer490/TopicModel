/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_METRICS_HPP
#define SIGTM_METRICS_HPP

#include "../sigtm.hpp"

namespace sigtm
{

template <class C1, class C2, class F>
uint set_intersection_num(C1&& c1, C2&& c2, bool is_sorted, F&& compare_func)
{
	if(!is_sorted){
		sig::sort(c1, compare_func);
		sig::sort(c2, compare_func);
	}
	
	uint ct = 0;
	auto it1 = std::begin(c1), ed1 = std::end(c1);
	auto it2 = std::begin(c2), ed2 = std::end(c2);
	
	while(it1 != ed1 && it2 != ed2){
		if(std::forward<F>(compare_func)(*it1, *it2)) ++it1;
		else if(std::forward<F>(compare_func)(*it2, *it1)) ++it2;
		else{
			++ct;
			++it1;
			++it2;
		}
	}
	
	return ct;
};

template <class MODEL>
struct Precision;

struct PrecisionBase
{
	template <class C1, class C2, class F
		//typename std::enable_if<sig::impl::container_traits<C1>::exist && sig::impl::container_traits<C2>::exist>::type*& = enabler
	>
	auto impl(C1&& estimates, C2&& answers, bool is_sorted, F&& compare_func) const->sig::Maybe<double>
	{
		return impl(estimates.size(), set_intersection_num(std::forward<C1>(estimates), std::forward<C2>(answers), is_sorted, std::forward<F>(compare_func)));
	}
	
	auto impl(uint estimate_num, uint intersection_num) const->sig::Maybe<double>
	{
		return estimate_num > 0 ? sig::Just(static_cast<double>(intersection_num) / estimate_num) : sig::Nothing<double>();
	}
};


template <class MODEL>
struct Recall;

struct RecallBase
{
	template <class C1, class C2, class F>
	auto impl(C1&& estimates, C2&& answers, bool is_sorted, F&& compare_func) const->sig::Maybe<double>
	{
		return impl(answers.size(), set_intersection_num(std::forward<C1>(estimates), std::forward<C2>(answers), is_sorted, std::forward<F>(compare_func)));
	}

	auto impl(uint answer_num, uint intersection_num) const->sig::Maybe<double>
	{
		return answer_num > 0 ? sig::Just(static_cast<double>(intersection_num) / answer_num) : sig::Nothing<double>();
	}
};


template <class MODEL>
struct F_Measure;

struct F_MeasureBase
{
	double impl(double precision, double recall) const
	{
		return (2 * precision * recall) / (precision + recall);
	}

	double operator()(double precision, double recall) const
	{
		return impl(precision, recall);
	}
};


}
#endif