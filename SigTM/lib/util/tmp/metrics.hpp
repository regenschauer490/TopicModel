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
uint set_intersection_num(C1 const& c1, C2 const& c2, bool is_sorted, F&& compare_func)
{
	if(!is_sorted){
		sig::sort(c1);
		sig::sort(c2);
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
	double impl(C1 const& estimates, C2 const& answers, bool is_sorted, F&& compare_func) const
	{
		return static_cast<double>(set_intersection_num(estimates, answers, is_sorted, std::forward<F>(compare_func))) / estimates.size();
	}
	
	double impl(uint estimate_num, uint intersection_num) const
	{
		return static_cast<double>(intersection_num) / estimate_num;
	}
}; 

}
#endif