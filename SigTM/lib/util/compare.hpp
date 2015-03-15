#include "../model/lda_gibbs.h"
#include "../model/lda_cvb.h"
#include "../model/mrlda.h"

namespace sigtm
{

/**
\brief
	@~japanese ベクトルや確率分布同士の類似度を測る	\n
	@~english Calculate similarity between feature vectors or distributions	\n
\detail
	@~japanese
	メソッドチェーンな感じに使用

	\tparam Select 比較を行うベクトルや分布．LDA::Distributionから選択
	\param lda 学習済のモデル
	\param id1,id2 比較対象のindex
	\return 比較関数を設定する関数オブジェクト

	@~english
	use like method-chain.
	the object returned from this function is function-object, and you can give it compare-method as template-parameter (see following example). 
	
	\tparam Select kinds of features you want compare．select from LDA::Distribution
	\param lda model which is finished training
	\param id1,id2 index pair you want to compare
	\return function-object to select compare-method

	@~
	\code
	auto lda = LDA_Gibbs::makeInstance(num_topics, inputdata, resume);
	lda.train(1000);

	auto sim_jsdiv = compare<LDA::Distribution::DOCUMENT>(lda, 0, 1).method<CompareMethodD::JS_DIV>();

	auto sim_cos = compare<LDA::Distribution::TERM_SCORE>(lda, 0, 1).method<CompareMethodV::COS>();

	if(sim_jsdiv){
		double sim = *sim_jsdiv;
	}
	if(sim_cos){
		double sim = *sim_cos;
	}
	\endcode
*/
template <LDA::Distribution Select>
auto compare(LDAPtr lda, Id id1, Id id2) ->typename LDA::Map2Cmp<Select>::type
{
	switch (lda->getDynamicType()) {
	case LDA::DynamicType::GIBBS:
		return std::static_pointer_cast<LDA_Gibbs>(lda)->compare<Select>(id1, id2);
	case LDA::DynamicType::CVB0:
		return std::static_pointer_cast<LDA_CVB0>(lda)->compare<Select>(id1, id2);
#if SIG_MSVC_ENV
	case LDA::DynamicType::MRLDA:
		return std::static_pointer_cast<MrLDA>(lda)->compare<Select>(id1, id2);
#endif
	default:
		assert(false);
	}
}
	
}
