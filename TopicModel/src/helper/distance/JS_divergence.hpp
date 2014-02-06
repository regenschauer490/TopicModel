#pragma once

#include "../../sigdm.hpp"

namespace sigdm{

//JS情報量
//条件：distribution[i] は正の値 かつ 総和が 1
//値域：[0, ∞)
template<template<class T, class = std::allocator<T>> class Container>
inline maybe<double> JS_Divergence(Container<double> const& distribution1, Container<double> const& distribution2)
{
	if(distribution1.size() != distribution2.size()) return nothing;

	const auto Log2 = [](double n)->double{ return log(n) / log(2); };

	const auto JS = [&](Container<double> const& d1, Container<double> const& d2)->double{
		double sum1 = 0, sum2 = 0;
		for(uint i=0; i<d1.size(); ++i){
			auto r = (d1[i] + d2[i]) * 0.5;
			sum1 += d1[i] * ( Log2(d1[i]) - Log2(r) );
			sum2 += d2[i] * ( Log2(d2[i]) - Log2(r) );
		}
		return (sum1 + sum2) * 0.5;
	};

	return JS(distribution1, distribution2);
}

}