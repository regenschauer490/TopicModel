#pragma once

#include "distance/cosine_similarity.hpp"
#include "distance/KL_divergence.hpp"
#include "distance/JS_divergence.hpp"

namespace sigdm{

//�x�N�g���̔�r��@
enum class CompareMethodV{ COS };

//�m�����z�̔�r��@
enum class CompareMethodD{ KL_DIV, JS_DIV };


//�ގ��x�s��
struct SimilarityMatrix{
	std::vector<std::vector<double>> data;
};
typedef std::shared_ptr <SimilarityMatrix const> SimilarityMatrixPtr;

//��ގ��x�s��
struct DisSimilarityMatrix{
	std::vector<std::vector<double>> data;
};
typedef std::shared_ptr <DisSimilarityMatrix const> DisSimilarityMatrixPtr;


//�e���f���N���X�̓����ɒ�`����C���i�[�N���X

//FUNC -> std::function< Container<double> (uint id) >
#define MakeCompareInnerClass(OUTER_CLASS) \
template <class FUNC>	\
class CmpD{	\
	uint _d1;	\
	uint _d2;	\
	FUNC _vp;	\
	bool _error;	\
\
public:	\
	CmpD(uint d1, uint d2, FUNC vector_producer, bool error = false) : _d1(d1), _d2(d2), _vp(vector_producer), _error(error){}	\
\
	maybe<double> Method(CompareMethodD method){\
		return _error	\
			? nothing	\
			: method == CompareMethodD::KL_DIV	\
				? KL_Divergence(_vp(_d1), _vp(_d2))	\
				: JS_Divergence(_vp(_d1), _vp(_d2))	\
		; }	\
};	\
\
template <class FUNC>	\
class CmpV{	\
	uint _d1;	\
	uint _d2;	\
	FUNC _vp;	\
	bool _error;	\
\
public:	\
	CmpV(uint d1, uint d2, FUNC vector_producer, bool error = false) : _d1(d1), _d2(d2), _vp(vector_producer), _error(error){}	\
\
	maybe<double> Method(CompareMethodV method){\
		return _error	\
			? nothing	\
			: method == CompareMethodV::COS	\
				? CosineSimilarity(_vp(_d1), _vp(_d2))	\
				: nothing	\
		; }	\
}


#define MakeDist2CmpMapBase \
template <enum class Distribution tag>	\
struct Map2Cmp{\
	typedef void type;	\
}


#define MakeDist2CmpMap(KEY, TYPE) \
template <>	\
struct Map2Cmp<KEY>{\
	typedef TYPE type;	\
}

}