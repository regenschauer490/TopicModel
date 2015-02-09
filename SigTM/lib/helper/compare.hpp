#include "../model/lda_gibbs.h"
#include "../model/lda_cvb.h"
#include "../model/mrlda.h"

namespace sigtm
{

// 確率分布同士の類似度を測る(メソッドチェーンな感じに使用)
template <LDA::Distribution Select>
auto compare(LDAPtr lda, Id id1, Id id2) ->typename LDA::Map2Cmp<Select>::type
{
	switch (lda->getDynamicType()) {
	case LDA::DynamicType::GIBBS:
		return std::static_pointer_cast<LDA_Gibbs>(lda)->compare<Select>(id1, id2);
	case LDA::DynamicType::MRLDA:
		return std::static_pointer_cast<MrLDA>(lda)->compare<Select>(id1, id2);
	case LDA::DynamicType::CVB0:
		return std::static_pointer_cast<LDA_CVB0>(lda)->compare<Select>(id1, id2);
	default:
		assert(false);
	}
}
	
}