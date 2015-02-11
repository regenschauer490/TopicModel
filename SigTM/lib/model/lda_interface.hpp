/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_LDA_INTERFACE_HPP
#define SIGTM_LDA_INTERFACE_HPP

#include "../sigtm.hpp"
#include "../helper/compare_method.hpp"
#include "../helper/data_format.hpp"

namespace sigtm
{

class LDA;
using LDAPtr = std::shared_ptr<LDA>;


template<class T> using MatrixTK = VectorT<VectorK<T>>;	// token - topic
template<class T> using MatrixDK = VectorD<VectorK<T>>;	// document - topic
template<class T> using MatrixVK = VectorV<VectorK<T>>;	// word - topic
template<class T> using MatrixKV = VectorK<VectorV<T>>;	// topic - word
/*
template<class T> using SVectorD = std::shared_ptr<VectorD<T>>;
template<class T> using SVectorK = std::shared_ptr<VectorK<T>>;
template<class T> using SVectorV = std::shared_ptr<VectorV<T>>;
template<class T> using SMatrixDK = SVectorD<SVectorK<T>>;
template<class T> using SMatrixVK = SVectorV<SVectorK<T>>;
template<class T> using SMatrixKV = SVectorK<SVectorV<T>>;
*/

#define SIG_INIT_VECTOR(type, index_name, value)\
	Vector ## index_name <type>(index_name ## _, value)

#define SIG_INIT_VECTOR_R(type, index_name, range, value)\
	Vector ## index_name <type>(range, value)

#define SIG_INIT_MATRIX(type, index_name1, index_name2, value)\
	Matrix ## index_name1 ## index_name2 <type>(index_name1 ## _, SIG_INIT_VECTOR(type, index_name2, value))

#define SIG_INIT_MATRIX_R(type, index_name1,range1, index_name2, range2, value)\
	Matrix ## index_name1 ## index_name2 <type>(range1, SIG_INIT_VECTOR_R(type, index_name2, range2, value))

#define SIG_INIT_MATRIX3(type, index_name1, index_name2, index_name3, value)\
	Vector ## index_name1 <type>(index_name1 ## _, SIG_INIT_MATRIX(type, index_name2, index_name3, value))


// LDAインタフェース
class LDA
{
public:
	enum class DynamicType{ GIBBS, MRLDA, CVB0 };		// 実装次第追加
	virtual DynamicType getDynamicType() const = 0;

public:
	// LDAで得られる確率分布やベクトル
	enum class Distribution{ DOCUMENT, TOPIC, TERM_SCORE };

	SIG_MakeCompareInnerClass(LDA);

	// method chain 生成
	SIG_MakeDist2CmpMapBase;
	SIG_MakeDist2CmpMap(Distribution::DOCUMENT, LDA::CmpD<std::function< VectorD<double>(DocumentId) >>);
	SIG_MakeDist2CmpMap(Distribution::TOPIC, LDA::CmpD<std::function< VectorK<double>(TopicId) >>);
	SIG_MakeDist2CmpMap(Distribution::TERM_SCORE, LDA::CmpV<std::function< VectorK<double>(TopicId) >>);
	//SIG_MakeDist2CmpMap(Distribution::DOC_TERM, LDA::CmpV);
	
	//template <LDA::Distribution Select>
	//auto compare(Id id1, Id id2) ->typename Map2Cmp<Select>::type;

protected:	
	// 確率分布同士の類似度を測る(メソッドチェーンな感じに使用)
	template <Distribution Select>
	auto compareDefault(Id id1, Id id2, uint D, uint K) const->typename Map2Cmp<Select>::type
	{
		return Select == Distribution::DOCUMENT
			? typename Map2Cmp<Select>::type(id1, id2, [this](DocumentId id){ return this->getTheta(id); }, id1 < D && id2 < D ? true : false)
			: Select == Distribution::TOPIC
			? typename Map2Cmp<Select>::type(id1, id2, [this](TopicId id){ return this->getPhi(id); }, id1 < K && id2 < K ? true : false)
			: Select == Distribution::TERM_SCORE
			? typename Map2Cmp<Select>::type(id1, id2, [this](TopicId id){ return this->getTermScore(id); }, id1 < K && id2 < K ? true : false)
			: typename Map2Cmp<Select>::type(id1, id2, [](TopicId id){ return std::vector<double>(); }, false);
	}

protected:
	LDA() = default;
	LDA(LDA const&) = delete;
	LDA(LDA&&) = delete;
	
public:
	virtual ~LDA(){}
	
	// モデルの学習を行う
	virtual void train(uint iteration_num) = 0;

	virtual void train(uint iteration_num, std::function<void(LDA const*)> callback) = 0;
		
	// コンソールに出力
	virtual void print(Distribution target) const = 0;

	// ファイルに出力
	// save_folder: 保存先のフォルダのパス
	// detail: 詳細なデータも全て出力するか
	virtual void save(Distribution target, FilepassString save_folder, bool detail = false) const = 0;

	//ドキュメントのトピック分布 [doc][topic]
	virtual auto getTheta() const->MatrixDK<double>;
	virtual auto getTheta(DocumentId d_id) const->VectorK<double> = 0;

	//トピックの単語分布 [topic][word]
	virtual auto getPhi() const->MatrixKV<double>;
	virtual auto getPhi(TopicId k_id) const->VectorV<double> = 0;

	//トピックを強調する単語スコア [topic][word]
	virtual auto getTermScore() const->MatrixKV<double> = 0;
	virtual auto getTermScore(TopicId t_id) const->VectorV<double> = 0;
	
	// 指定トピックの上位return_word_num個の、語彙とスコアを返す
	// [topic][ranking]<vocab, score>
	virtual auto getWordOfTopic(Distribution target, uint return_word_num) const->VectorK< std::vector< std::tuple<std::wstring, double>>>;
	// [ranking]<vocab, score>
	virtual auto getWordOfTopic(Distribution target, uint return_word_num, TopicId k_id) const->std::vector< std::tuple<std::wstring, double>> = 0;

	// 指定ドキュメントの上位return_word_num個の、語彙とスコアを返す
	// [doc][ranking]<vocab, score>
	virtual auto getWordOfDocument(uint return_word_num) const->VectorD< std::vector< std::tuple<std::wstring, double>>>;
	//[ranking]<vocab, score>
	virtual auto getWordOfDocument(uint return_word_num, DocumentId d_id) const->std::vector< std::tuple<std::wstring, double>> = 0;

	virtual uint getDocumentNum() const = 0;
	virtual uint getTopicNum() const = 0;
	virtual uint getWordNum() const = 0;

	// get hyper-parameter of topic distribution
	virtual auto getAlpha() const->VectorK<double> = 0;
	// get hyper-parameter of word distribution
	virtual auto getBeta() const->VectorV<double> = 0;

	virtual double getLogLikelihood() const = 0;

	virtual double getPerplexity() const = 0;
};

const std::function<void(LDA const*)> null_lda_callback = [](LDA const*){};


inline auto LDA::getTheta() const->MatrixDK<double>
{
	MatrixDK<double> theta;

	for (DocumentId d = 0, D = getDocumentNum(); d < D; ++d) theta.push_back(getTheta(d));

	return theta;
}

inline auto LDA::getPhi() const->MatrixKV<double>
{
	MatrixKV<double> phi;

	for (TopicId k = 0, K = getTopicNum(); k < K; ++k) phi.push_back(getPhi(k));

	return std::move(phi);
}

inline auto LDA::getWordOfTopic(LDA::Distribution target, uint return_word_num) const->VectorK< std::vector< std::tuple<std::wstring, double>>>
{
	VectorK< std::vector< std::tuple<std::wstring, double> > > result;

	for (TopicId k = 0, K = getTopicNum(); k < K; ++k){
		result.push_back(getWordOfTopic(target, return_word_num, k));
	}

	return std::move(result);
}

inline auto LDA::getWordOfDocument(uint return_word_num) const->VectorD< std::vector< std::tuple<std::wstring, double>>>
{
	std::vector< std::vector< std::tuple<std::wstring, double>>> result;

	for (DocumentId d = 0, D = getDocumentNum(); d < D; ++d){
		result.push_back(getWordOfDocument(return_word_num, d));
	}

	return std::move(result);
}

}
#endif
