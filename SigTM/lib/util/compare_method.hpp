/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_COMPARE_METHOD_HPP
#define SIG_COMPARE_METHOD_HPP

#include "SigUtil/lib/distance/cosine_similarity.hpp"
#include "SigUtil/lib/distance/KL_divergence.hpp"
#include "SigUtil/lib/distance/JS_divergence.hpp"

namespace sigtm{

/**
\brief
	ベクトルの比較手法 \ref g_compare_similarity
*/
enum class CompareMethodV{
	COS	/**< @~japanese コサイン類似度 @~english cosine similarity */
};

/**
\brief
	確率分布の比較手法 \ref g_compare_similarity
*/
enum class CompareMethodD{
	KL_DIV,	/**< @~japanese KL情報量 @~english KL Divergence */
	JS_DIV	/**< @~japanese JS情報量 @~english JS Divergence */
};

// todo: enum class -> functor


// 各モデルのクラス内部に定義するインナークラス
//FUNC :: std::function< Container<double> (uint id) >

#define SIG_MakeCompareInnerClass(OUTER_CLASS) \
template <class FUNC>	\
class CmpD{	\
	uint d1_;	\
	uint d2_;	\
	FUNC vp_;	\
	bool valid_;	\
\
public:	\
	CmpD(uint d1, uint d2, FUNC vector_producer, bool valid = true) : d1_(d1), d2_(d2), vp_(vector_producer), valid_(valid){}	\
\
	template <CompareMethodD Select>\
	Maybe<double> method(){\
		return !valid_	\
			? Nothing<double>()	\
			: Select == CompareMethodD::KL_DIV	\
				? sig::kl_divergence(vp_(d1_), vp_(d2_))	\
				: sig::js_divergence(vp_(d1_), vp_(d2_))	\
		; }	\
};	\
\
template <class FUNC>	\
class CmpV{	\
	uint d1_;	\
	uint d2_;	\
	FUNC vp_;	\
	bool valid_;	\
\
public:	\
	CmpV(uint d1, uint d2, FUNC vector_producer, bool valid = false) : d1_(d1), d2_(d2), vp_(vector_producer), valid_(valid){}	\
\
	template <CompareMethodV Select>\
	Maybe<double> method(){\
		return !valid_	\
			? Nothing<double>()	\
			: Select == CompareMethodV::COS	\
				? Just(sig::cosine_similarity(vp_(d1_), vp_(d2_)))	\
				: Nothing<double>()	\
		; }	\
};


#define SIG_MakeDist2CmpMapBase \
template <Distribution tag, class = void>	\
struct Map2Cmp{\
	using type = void;	\
};

#define SIG_MakeDist2CmpMap(KEY, TYPE) \
template <class D>	\
struct Map2Cmp<KEY, D>{\
	using type = TYPE;	\
};

}
#endif
