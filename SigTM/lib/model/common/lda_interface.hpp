/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_LDA_INTERFACE_HPP
#define SIGTM_LDA_INTERFACE_HPP

#include "../../sigtm.hpp"
#include "../../util/compare_method.hpp"
#include "../../data/data_format.hpp"

namespace sigtm
{

class LDA;
using LDAPtr = std::shared_ptr<LDA>;

template<class T> using MatrixTK = VectorT<VectorK<T>>;	// token - topic
template<class T> using MatrixDK = VectorD<VectorK<T>>;	// document - topic
template<class T> using MatrixVK = VectorV<VectorK<T>>;	// word - topic
template<class T> using MatrixKV = VectorK<VectorV<T>>;	// topic - word


/// LDAインタフェース
class LDA
{
public:
	/// サブクラスとして利用できる実装の種類
	enum class DynamicType{ GIBBS, CVB0, MRLDA };

	virtual DynamicType getDynamicType() const = 0;

public:
	/**
	\brief
		@~japanese LDAの学習結果として得られる確率分布やスコアの種類	\n
		@~english probability distributions and scores gained from trained model	\n

	\details
		@~ see details in \ref g_model_param , \ref g_term_score
	*/
	enum class Distribution{
		DOCUMENT,	/**< @~japanese 文書が持つトピック比率θを指す @~english indicate the θ */
		TOPIC,		/**< @~japanese トピックが持つ語彙比率Φを指す @~english indicate the Φ */
		TERM_SCORE	/**< @~japanese トピックを強調する語彙スコアを指す @~english indicate the word score of topic */
	};

	SIG_MakeCompareInnerClass(LDA);

	// method chain 生成
	SIG_MakeDist2CmpMapBase;
	SIG_MakeDist2CmpMap(Distribution::DOCUMENT, LDA::CmpD<std::function< VectorD<double>(DocumentId) >>);
	SIG_MakeDist2CmpMap(Distribution::TOPIC, LDA::CmpD<std::function< VectorK<double>(TopicId) >>);
	SIG_MakeDist2CmpMap(Distribution::TERM_SCORE, LDA::CmpV<std::function< VectorK<double>(TopicId) >>);

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
	virtual ~LDA() = default;
	
	/**
	\brief
		@~japanese モデルの学習を行う
		@~english Train model

	\details
		@~japanese
		\param num_iteration 学習の反復回数（全トークンの変分パラメータの更新を1反復とする）

		@~english
		\param num_iteration number of iterations (define updating all variational parameters as one iteration)
	*/
	virtual void train(uint num_iteration) = 0;

	/**
	\brief
		@~japanese モデルの学習を行う
		@~english Train model

	\details
		@~japanese
		\param num_iteration 学習の反復回数（全トークンの変分パラメータの更新を1反復とする）
		\param call_back 毎回の反復終了時に行う処理

		@~english
		\param num_iteration number of iterations (define updating all variational parameters as one iteration)
		\param call_back user-defined processing to be done at each iteration end
	*/
	virtual void train(uint num_iteration, std::function<void(LDA const*)> callback) = 0;
		
	/**
	\brief
		@~japanese コンソールに出力
		@~english Output to console

	\details
		@~japanese
		\param target 出力したいパラメータやスコア

		@~english
		\param target parameter and score you want to output
	*/
	virtual void print(Distribution target) const = 0;

	/**
	\brief
		@~japanese ファイルに出力
		@~english Output to file

	\details
		@~japanese
		\param target 出力したいパラメータやスコア
		\param save_dir 出力先のパス
		\param detail true:すべての値を出力, false:特徴的な値のみを出力

		@~english
		\param target parameter and score you want to output
		\param save_dir output directory
		\param detail true: output all values, false: output only some characteristic values
	*/
	virtual void save(Distribution target, FilepassString save_dir, bool detail = false) const = 0;

	/**
	\brief
		@~japanese 文書が持つトピック比率を取得
		@~english Get the topic proportion of each document

	\details
		@~
		\return θ (:: [doc][topic])
	*/
	virtual auto getTheta() const->MatrixDK<double>;

	/**
	\brief
		@~japanese 指定文書が持つトピック比率を取得
		@~english Get the topic proportion of specified document

	\details
		@~japanese
		\param d_id 取得したい文書ID
		\return θ_d (:: [topic])

		@~english
		\param d_id document id you want to get
		\return θ_d (:: [topic])
	*/
	virtual auto getTheta(DocumentId d_id) const->VectorK<double> = 0;

	/**
	\brief
		@~japanese トピックが持つ語彙比率を取得
		@~english Get the word proportion of each topic

	\details
		@~
		\return Φ (:: [topic][word])
	*/
	virtual auto getPhi() const->MatrixKV<double>;

	/**
	\brief
		@~japanese 指定トピックが持つ語彙比率を取得
		@~english Get the word proportion of specified topic

	\details
		@~japanese
		\param k_id 取得したいトピックID
		\return Φ_k (:: [word])

		@~english
		\param k_id topic id you want to get
		\return Φ_k (:: [word])
	*/
	virtual auto getPhi(TopicId k_id) const->VectorV<double> = 0;

	/**
	\brief
		@~japanese トピックを強調する語彙スコアを取得
		@~english Get the word-scores of each topic

	\details
		@~
		(\ref g_term_score)

		\return term-score (:: [topic][word])
	*/
	virtual auto getTermScore() const->MatrixKV<double> = 0;

	/**
	\brief
		@~japanese 指定トピックを強調する語彙スコアを取得	\n
		@~english Get the word-scores of specified topic	\n

	\details
		@~japanese
		(\ref g_term_score)

		\param k_id 取得したいトピックID
		\return term-score_t (:: [word])

		@~english
		(\ref g_term_score)

		\param k_id topic id you want to get
		\return term-score_t (:: [word])
	*/
	virtual auto getTermScore(TopicId k_id) const->VectorV<double> = 0;
	
	/**
	\brief
		@~japanese トピックの代表語彙とそのスコアを取得	\n
		@~english Get the characteristic words and their scores of each topic	\n

	\details
		@~japanese
		各トピックにおいて，targetで指定した語彙ベクトルを降順にソートし，その上位num_get_words個の語彙とスコアを取得

		\param target 語彙スコアとして利用するパラメータ
		\param num_get_words 上位何個までの語彙を取得するか
		\return result (:: [topic][ranking]<vocabulary, score>)

		@~english
		In each topic, make word-ranking based on the scores which you select of target.	\n
		Then select top num_get_words from this ranking, and make pair of <vocabulary(word), score>.

		\param target the scores of this target are used to make word-ranking
		\param num_get_words the number of words from top
		\return result (:: [topic][ranking]<vocabulary, score>)
	*/
	virtual auto getWordOfTopic(Distribution target, uint num_get_words) const->VectorK< std::vector< std::tuple<std::wstring, double>>>;
	
	/**
	\brief
		@~japanese 指定トピックの代表語彙とそのスコアを取得	\n
		@~english Get the characteristic words and their scores of specific topic	\n

	\details
		@~japanese
		IDを指定したトピックにおいて，targetで指定した語彙ベクトルを降順にソートし，その上位num_get_words個の語彙とスコアを取得

		\param target 語彙スコアとして利用するパラメータの選択
		\param num_get_words 上位何個までの語彙を取得するか
		\param k_id 取得したいトピックID
		\return result (:: [ranking]<vocab, score>)

		@~english
		In specific topic, make word-ranking based on the scores which you select of target.	\n
		Then select top num_get_words from this ranking, and make pair of <vocabulary(word), score>.

		\param target the scores of this target are used to make word-ranking
		\param num_get_words the number of words from top
		\param k_id topic id you want to get
		\return result (:: [ranking]<vocabulary, score>)
	*/
	virtual auto getWordOfTopic(Distribution target, uint num_get_words, TopicId k_id) const->std::vector< std::tuple<std::wstring, double>> = 0;

	/**
	\brief
		@~japanese 文書の代表語彙とそのスコアを取得	\n
		@~english get the characteristic words and their scores of each document	\n

	\details
		@~japanese
		各文書において，targetで指定した語彙ベクトルを利用して生成確率( P(d,w) = Σ_k P(w|z,Φ) * P(z|θ_d) * P(Φ|β) * P(θ_d|α) )が高い順に語彙をソートし，その上位num_get_words個の語彙とスコアを取得

		\param target 語彙スコアとして利用するパラメータ
		\param num_get_words 上位何個までの語彙を取得するか
		\return result (:: [document][ranking]<vocabulary, score>)

		@~english
		In each document, make word-ranking based on the word-generative probability ( P(d,w) = Σ_k P(w|z,Φ) * P(z|θ_d) * P(Φ|β) * P(θ_d|α) ).	\n
		Then select top num_get_words from this ranking, and make pair of <vocabulary(word), score>.

		\param target the scores of this target are used to make word-ranking
		\param num_get_words the number of words from top
		\return result (:: [document][ranking]<vocabulary, score>)
	*/
	virtual auto getWordOfDocument(Distribution target, uint num_get_words) const->VectorD< std::vector< std::tuple<std::wstring, double>>>;
	
	/**
	\brief
		@~japanese 指定文書の代表語彙とそのスコアを取得	\n
		@~english get the characteristic words and their scores of specified document	\n

	\details
		@~japanese
		IDを指定した文書において，targetで指定した語彙ベクトルを利用して生成確率( P(d,w) = Σ_k P(w|z,Φ) * P(z|θ_d) * P(Φ|β) * P(θ_d|α) )が高い順に語彙をソートし，その上位num_get_words個の語彙とスコアを取得

		\param target スコアとして利用するパラメータの選択
		\param num_get_words 上位何個までの語彙を取得するか
		\param d_id 取得したい文書ID
		\return result (:: [ranking]<vocabulary, score>)

		@~english
		In specific document, make word-ranking based on the word-generative probability ( P(d,w) = Σ_k P(w|z,Φ) * P(z|θ_d) * P(Φ|β) * P(θ_d|α) ).	\n
		Then select top num_get_words from this ranking, and make pair of <vocabulary(word), score>.

		\param target the scores of this target are used to make word-ranking
		\param num_get_words the number of words from top
		\param d_id document id you want to get
		\return result (:: [ranking]<vocabulary, score>)
	*/
	virtual auto getWordOfDocument(Distribution target, uint num_get_words, DocumentId d_id) const->std::vector< std::tuple<std::wstring, double>> = 0;

	/**
	\brief
		@~japanese 文書数を取得
		@~english Get the number of documents
	*/
	virtual uint getDocumentNum() const = 0;

	/**
	\brief
		@~japanese トピック数を取得
		@~english Get the number of topics
	*/
	virtual uint getTopicNum() const = 0;

	/**
	\brief
		@~japanese 語彙数を取得
		@~english Get the number of words (vocabularies)
	*/
	virtual uint getWordNum() const = 0;

	/**
	\brief
		@~japanese ハイパーパラメータαを取得（\ref g_hparam_alpha ）
		@~english Get \ref g_hparam_alpha
	*/
	virtual auto getAlpha() const->VectorK<double> = 0;
	
	/**
	\brief
		@~japanese ハイパーパラメータβを取得（\ref g_hparam_beta ）
		@~english Get \ref g_hparam_beta
	*/
	virtual auto getBeta() const->VectorV<double> = 0;

	/**
	\brief
		@~japanese モデルの対数尤度（\ref g_log_likelihood ）を取得
		@~english Get model \ref g_log_likelihood
	*/
	virtual double getLogLikelihood() const = 0;

	/**
	\brief
		@~japanese モデルの \ref g_perplexity を取得
		@~english Get model \ref g_perplexity
	*/
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

inline auto LDA::getWordOfTopic(LDA::Distribution target, uint num_get_words) const->VectorK< std::vector< std::tuple<std::wstring, double>>>
{
	VectorK< std::vector< std::tuple<std::wstring, double> > > result;

	for (TopicId k = 0, K = getTopicNum(); k < K; ++k){
		result.push_back(getWordOfTopic(target, num_get_words, k));
	}

	return std::move(result);
}

inline auto LDA::getWordOfDocument(LDA::Distribution target, uint num_get_words) const->VectorD< std::vector< std::tuple<std::wstring, double>>>
{
	std::vector< std::vector< std::tuple<std::wstring, double>>> result;

	for (DocumentId d = 0, D = getDocumentNum(); d < D; ++d){
		result.push_back(getWordOfDocument(target, num_get_words, d));
	}

	return std::move(result);
}


// 初期化用
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


}
#endif
