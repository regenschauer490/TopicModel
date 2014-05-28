/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_DISTANCE_H
#define SIG_DISTANCE_H

#include "SigUtil/lib/functional.hpp"

#undef max
#undef min

namespace sigtm{

	//ミンコフスキー距離
	template <size_t P>
	struct MinkowskiDistance
	{
		template <class C1, class C2>
		double operator()(C1 const& vec1, C2 const& vec2) const
		{
			using T = typename std::common_type<typename sig::container_traits<C1>::value_type, typename sig::container_traits<C2>::value_type>::type;
			
			return std::pow(
				std::inner_product(std::begin(vec1), std::end(vec1), std::begin(vec2), T(), std::plus<T>(), [&](T v1, T v2){ return pow(sig::DeltaAbs(v1, v2), P); }),
				1.0 / P
			);
		}
	};

	//マンハッタン距離
	using ManhattanDistance = MinkowskiDistance<1>;

	const ManhattanDistance manhattan_distance;

	//ユークリッド距離
	using EuclideanDistance = MinkowskiDistance<2>;

	const EuclideanDistance euclidean_distance;

	//キャンベラ距離
	struct CanberraDistance
	{
		template <class C1, class C2>
		double operator()(C1 const& vec1, C2 const& vec2) const
		{
			using T = typename std::common_type<typename sig::container_traits<C1>::value_type, typename sig::container_traits<C2>::value_type>::type;
			
			auto tmp = sig::ZipWith([](T val1, T val2){ return static_cast<T>(abs(val1 - val2)) / (abs(val1) + abs(val2)); }, vec1, vec2);

			return std::accumulate(std::begin(tmp), std::end(tmp), T(), std::plus<T>());
		}
	};

	const CanberraDistance canberra_distance;

	//バイナリ距離
	struct BinaryDistance
	{
		template <class C1, class C2,
			typename = typename std::enable_if<std::is_same<typename container_traits<C1>::value_type, int>::value || std::is_same<typename container_traits<C1>::value_type, bool>::value>::type,
			typename = typename std::enable_if<std::is_same<typename container_traits<C2>::value_type, int>::value || std::is_same<typename container_traits<C1>::value_type, bool>::value>::type
		>
		double operator()(C1 const& vec1, C2 const& vec2) const
		{
			int ether = 0, both = 0;
			for (auto it1 = std::begin(vec1), it2 = std::begin(vec2), end1 = std::end(vec1), end2 = std::end(vec2); it1 != end1 && it2 != end2; ++it1, ++it2){
				if (*it1 == 1 && *it2 == 1) ++both;
				else if (*it1 == 1 || *it2 == 1) ++ether;
			}
			return static_cast<double>(ether) / (ether + both);
		}
	};

	const BinaryDistance binary_distance;
}
#endif