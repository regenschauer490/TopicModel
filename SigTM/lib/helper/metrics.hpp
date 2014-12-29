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

struct PrecisionImpl
{
	// estimates, answers: ����E����id�W��
	// is_sorted: id���Ƀ\�[�g����Ă��邩
	// compare_func: id���m�̑召��r���s���֐�
	template <class C1, class C2, class F
		//typename std::enable_if<sig::impl::container_traits<C1>::exist && sig::impl::container_traits<C2>::exist>::type*& = enabler
	>
	auto impl(C1&& estimates, C2&& answers, bool is_sorted, F&& compare_func) const->sig::Maybe<double>
	{
		return impl(estimates.size(), set_intersection_num(std::forward<C1>(estimates), std::forward<C2>(answers), is_sorted, std::forward<F>(compare_func)));
	}
	
	auto impl(uint estimate_num, uint intersection_num) const->sig::Maybe<double>
	{
		return estimate_num ? sig::Just(static_cast<double>(intersection_num) / estimate_num) : sig::Nothing<double>();
	}
};


template <class MODEL>
struct Recall;

struct RecallImpl
{
	// estimates, answers: ����E����id�W��
	// is_sorted: id���Ƀ\�[�g����Ă��邩
	// compare_func: id���m�̑召��r���s���֐�
	template <class C1, class C2, class F>
	auto impl(C1&& estimates, C2&& answers, bool is_sorted, F&& compare_func) const->sig::Maybe<double>
	{
		return impl(answers.size(), set_intersection_num(std::forward<C1>(estimates), std::forward<C2>(answers), is_sorted, std::forward<F>(compare_func)));
	}

	auto impl(uint answer_num, uint intersection_num) const->sig::Maybe<double>
	{
		return answer_num ? sig::Just(static_cast<double>(intersection_num) / answer_num) : sig::Nothing<double>();
	}
};


template <class MODEL>
struct F_Measure;

struct F_MeasureImpl
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


template <class MODEL>
struct AveragePrecision;

struct AveragePrecisionImpl
{
	// rankings: ���E�����L���Oid
	// answers: ����id�W��
	// is_answer_sorted: id���Ƀ\�[�g����Ă��邩
	// compare_func: id���m�̑召��r���s���֐�
	template <class C1, class C2>
	auto impl(C1&& rankings, C2&& answers) const->sig::Maybe<double>
	{
		double sum = 0;
		uint ct = 0;
		
		auto end = std::end(answers);

		for (uint r = 0, size = rankings.size(); r < size; ++r) {
			auto f = std::find(std::begin(answers), std::end(answers), rankings[r]);

			if (f != end) {
				++ct;
				sum += static_cast<double>(ct) / (r+1);
				//std::cout << r+1 << ": " << static_cast<double>(ct) / (r + 1) << std::endl;
			}
		}

		return ct ? sig::Just(sum / ct) : sig::Nothing<double>();
	}

	template <class C1, class C2, class D1, class D2>
	auto impl(C1&& rankings, C2&& answers, D1&& dumy1, D2&& dummy2) const->sig::Maybe<double>
	{
		return impl(std::forward<C1>(rankings), std::forward<C2>(answers));
	}
};


template <class MODEL>
struct CatalogueCoverage;

struct CatalogueCoverageImpl
{
	// estimate_sets: ����id�W���̃R���e�i
	// total_num: �Sid��
	template <class CC,
		class C = typename sig::impl::container_traits<typename sig::impl::remove_const_reference<CC>::type>::value_type,
		class T = typename sig::impl::container_traits<C>::value_type
	>
	auto impl(CC&& estimate_sets, uint total_num) const->sig::Maybe<double>
	{
		std::unordered_set<T> result;

		for (auto& est_set : estimate_sets) {
			for(auto e : est_set) result.emplace(e);
		}

		return result.size() / static_cast<double>(total_num);
	}
};

}
#endif