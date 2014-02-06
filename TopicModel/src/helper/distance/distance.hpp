#pragma once

#include "../../sigdm.hpp"

#undef min

namespace sigdm{

	//ミンコフスキー距離
	template <std::size_t P>
	struct MinkowskiDistance
	{
		template < class T, template < class T_, class = std::allocator <T_ >> class Container>
		static double f(Container<T> const& vec1, Container<T> const& vec2){
			return pow(
				sig::Accumulate(
					sig::ZipWith<double>(vec1, vec2, [](T val1, T val2){ return pow(abs(val1 - val2), P); }),
					0,
					std::plus<double>()
				),
				1.0 / P
				);
		}
	};

	//マンハッタン距離
	typedef MinkowskiDistance<1> ManhattanDistance;

	//ユークリッド距離
	typedef MinkowskiDistance<2> EuclideanDistance;

	//キャンベラ距離
	struct CanberraDistance
	{
		template < class T, template < class T_, class = std::allocator <T_ >> class Container>
		static double f(Container<T> const& vec1, Container<T> const& vec2){
			return sig::Accumulate(
				sig::ZipWith<double>(vec1, vec2, [](T val1, T val2){ return static_cast<double>(abs(val1 - val2)) / (abs(val1) + abs(val2)); }),
				0,
				std::plus<double>()
				);
		}
	};

	//バイナリ距離
	struct BinaryDistance
	{
		template < template < class T_, class = std::allocator <T_ >> class Container>
		static double f(Container<int> const& vec1, Container<int> const& vec2){
			int ether = 0, both = 0;
			for (auto it1 = vec1.begin(), it2 = vec2.begin(), end1 = vec1.end(), end2 = vec2.end(); it1 != end1 && it2 != end2; ++it1, ++it2){
				if (*it1 == 1 && *it2 == 1) ++both;
				else if (*it1 == 1 || *it2 == 1) ++ether;
			}
			return static_cast<double>(ether) / (ether + both);
		}

		template < template < class T_, class = std::allocator <T_ >> class Container>
		static double f(Container<bool> const& vec1, Container<bool> const& vec2){
			int ether = 0, both = 0;
			for (auto it1 = vec1.begin(), it2 = vec2.begin(), end1 = vec1.end(), end2 = vec2.end(); it1 != end1 && it2 != end2; ++it1, ++it2){
				if (*it1 && *it2) ++both;
				else if (*it1 || *it2) ++ether;
			}
			return static_cast<double>(ether) / (ether + both);
		}
	};
}