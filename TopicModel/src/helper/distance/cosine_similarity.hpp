#pragma once

#include <numeric>
#include "../../sigdm.hpp"

namespace sigdm{

//コサイン類似度
//値域：[-1, 1]
template<class T, template<class T, class = std::allocator<T>> class Container>
inline maybe<double> CosineSimilarity(Container<T> const& vector1, Container<T> const& vector2)
{
	if(vector1.size() != vector2.size()) return nothing;

	const auto Abs = [](Container<T> const& vec)->double{
		return std::sqrt( accumulate(vec.begin(), vec.end(), static_cast<T>(0), [](T sum, T val){ return sum + val*val; }) );
	};

	return std::inner_product(vector1.begin(), vector1.end(), vector2.begin(), static_cast<T>(0)) / (Abs(vector1) * Abs(vector2));
}

}