/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NORM_H
#define SIG_NORM_H

#include <numeric>
#undef max
#undef min

namespace sigtm{

template <size_t P, class C>
double Norm(C const& vec)
{
	using T = typename sig::container_traits<C>::value_type;

	return std::pow(
		std::accumulate(std::begin(vec), std::end(vec), static_cast<T>(0), [&](T sum, T val){ return sum + std::pow(val, P); }),
		1.0 / P
	);
}

template <class C>
double L1Norm(C const& vec)
{
	return Norm<1>(vec);
}

template <class C>
double L2Norm(C const& vec)
{
	return Norm<2>(vec);
}

}
#endif