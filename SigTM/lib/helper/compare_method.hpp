/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_COMPARE_METHOD_HPP
#define SIG_COMPARE_METHOD_HPP

#include "distance/cosine_similarity.hpp"
#include "distance/KL_divergence.hpp"
#include "distance/JS_divergence.hpp"

namespace sigtm{

//ベクトルの比較手法
enum class CompareMethodV{ COS };

//確率分布の比較手法
enum class CompareMethodD{ KL_DIV, JS_DIV };


//類似度行列
struct SimilarityMatrix{
	std::vector<std::vector<double>> data;
};
typedef std::shared_ptr <SimilarityMatrix const> SimilarityMatrixPtr;

//非類似度行列
struct DisSimilarityMatrix{
	std::vector<std::vector<double>> data;
};
typedef std::shared_ptr <DisSimilarityMatrix const> DisSimilarityMatrixPtr;


//各モデルクラスの内部に定義するインナークラス

//FUNC -> std::function< Container<double> (uint id) >
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
	maybe<double> method(CompareMethodD method){\
		return !valid_	\
			? nothing	\
			: method == CompareMethodD::KL_DIV	\
				? kl_divergence(vp_(d1_), vp_(d2_))	\
				: js_divergence(vp_(d1_), vp_(d2_))	\
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
	CmpV(uint d1, uint d2, FUNC vector_producer, bool error = false) : d1_(d1), d2_(d2), vp_(vector_producer), valid_(valid){}	\
\
	maybe<double> method(CompareMethodV method){\
		return !valid_	\
			? nothing	\
			: method == CompareMethodV::COS	\
				? cosine_similarity(vp_(d1_), vp_(d2_))	\
				: nothing	\
		; }	\
}


#define SIG_MakeDist2CmpMapBase \
template <enum class Distribution tag>	\
struct Map2Cmp{\
	typedef void type;	\
}


#define SIG_MakeDist2CmpMap(KEY, TYPE) \
template <>	\
struct Map2Cmp<KEY>{\
	typedef TYPE type;	\
}

}
#endif