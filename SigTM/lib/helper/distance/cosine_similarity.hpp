﻿/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_COSINE_SIMILARITY_H
#define SIG_COSINE_SIMILARITY_H

#include <numeric>
#include "norm.hpp"
#include "SigUtil/lib/sigutil.hpp"

namespace sigtm{

struct CosineSimilarity
{
	//コサイン類似度
	//値域：[-1, 1]
	//失敗時：boost::none (if not use boost, return 0)
	template<class C1, class C2>
	auto operator()(C1 const& vec1, C2 const& vec2) const ->typename sig::Just<double>::type
	{
		using T = std::common_type<typename sig::container_traits<C1>::value_type, typename sig::container_traits<C2>::value_type>::type;

		if(vec1.size() != vec2.size()) return sig::Nothing(0);

		return typename sig::Just<double>::type(std::inner_product(std::begin(vec1), std::end(vec1), std::begin(vec2), static_cast<T>(0)) / (L2Norm(vec1) * L2Norm(vec2)));
	}
};

const CosineSimilarity cosine_similarity;

}
#endif