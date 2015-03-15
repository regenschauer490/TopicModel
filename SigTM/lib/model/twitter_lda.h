/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_TWITTER_LDA_H
#define SIGTM_TWITTER_LDA_H

#include "common/lda_module.hpp"
#include "SigUtil/lib/container/array.hpp"

namespace sigtm
{

template<class T> using VectorB = sig::array<T, 2>;	// for bernoulli parameter
template<class T> using MatrixUD = VectorU<VectorD<T>>;	// user - tweet
template<class T> using MatrixUK = VectorU<VectorK<T>>;	// user - topic
template<class T> using MatrixUB = VectorU<VectorB<T>>;	// user - choice(topic or background)
template<class T> using MatrixUDT = VectorU<VectorD<VectorT<T>>>;	// user - tweet - token

class TwitterLDA;
using TwitterLDAPtr = std::shared_ptr<TwitterLDA>;

const std::function<void(TwitterLDA const*)> null_twlda_callback = [](TwitterLDA const*){};

/// Twitter-LDA (estimate by Gibbs Sampling)
/**
*/
class TwitterLDA final : private impl::LDA_Module
{
	DocumentSetPtr input_data_;
	TokenList const& tokens_;	// ※ソート操作による元データ変更の可能性あり

	const uint U_;				// number of users
	const uint K_;				// number of topics
	const uint V_;				// number of words
	const VectorU<UserId> D_;	// number of tweets in each user
	const MatrixUD<Id> T_;		// number of tokens in each tweet

	VectorK<double> alpha_;			// dirichlet hyper parameter of theta
	VectorV<double> beta_;			// dirichlet hyper parameter of phi
	VectorB<double> gamma_;			// dirichlet hyper parameter of pi

	MatrixUK<uint> user_ct_;		// tweet count of each topic in each user
	MatrixVK<uint> word_ct_;		// token count of each topic in each word ([word][K_] is word count of background)
	VectorK<uint> topic_ct_;		// token count of each topic
	VectorB<uint> y_ct_;			// token count of both y ([0]:background, [1]:topic)

	MatrixUD<uint> z_;				// topic assigned to each token
	MatrixUDT<bool> y_;				// choice between background words and topic words

	double alpha_sum_;
	double beta_sum_;
	VectorK<double> tmp_p_;
	MatrixKV<double> term_score_;	// word score of emphasizing each topic
	uint total_iter_ct_;

	sig::SimpleRandom<uint> rand_ui_;
	sig::SimpleRandom<double> rand_d_;

	using TokenIter = TokenList::const_iterator;

public:
	// LDAで得られる確率分布やベクトル
	enum class Distribution{ USER, TWEET, TOPIC, TERM_SCORE };

	SIG_MakeCompareInnerClass(TwitterLDA);

protected:
	// method chain 生成
	SIG_MakeDist2CmpMapBase;
	SIG_MakeDist2CmpMap(Distribution::USER, TwitterLDA::CmpD<std::function< VectorU<double>(UserId) >>);
	SIG_MakeDist2CmpMap(Distribution::TWEET, TwitterLDA::CmpD<std::function< VectorD<double>(DocumentId) >>);
	SIG_MakeDist2CmpMap(Distribution::TOPIC, TwitterLDA::CmpD<std::function< VectorK<double>(TopicId) >>);
	SIG_MakeDist2CmpMap(Distribution::TERM_SCORE, TwitterLDA::CmpV<std::function< VectorK<double>(TopicId) >>);
	
private:
	TwitterLDA() = delete;
	TwitterLDA(TwitterLDA const&) = delete;

	TwitterLDA(bool resume, uint num_topics, DocumentSetPtr input_data, Maybe<VectorK<double>> alpha, Maybe<VectorV<double>> beta, Maybe<VectorB<double>> gamma) :
		input_data_(input_data), tokens_(input_data->tokens_), U_(input_data->getDocNum()), K_(num_topics), V_(input_data->getWordNum()),
		alpha_(isJust(alpha) ? fromJust(alpha) : SIG_INIT_VECTOR(double, K, default_alpha_base / K_)), beta_(isJust(beta) ? fromJust(beta) : SIG_INIT_VECTOR(double, V, default_beta)), gamma_(isJust(gamma) ? fromJust(gamma) : VectorB<double>{ 0.5, 0.5 }),
		user_ct_(SIG_INIT_MATRIX(uint, U, K, 0)), word_ct_(SIG_INIT_MATRIX_R(uint, V, V_, K, K_+1, 0)), topic_ct_(SIG_INIT_VECTOR(uint, K, 0)),
		tmp_p_(SIG_INIT_VECTOR(double, K, 0)), term_score_(SIG_INIT_MATRIX(double, K, V, 0)), total_iter_ct_(0),
		rand_ui_(0, K_ - 1, FixedRandom), rand_d_(0.0, 1.0, FixedRandom)
	{
		init(resume);
	}
	
	void init(bool resume);
	void saveResumeData() const;
	void updateY(Token const& token, const uint t_pos);
	void updateZ(TokenIter begin, TokenIter end);
	
public:
	~TwitterLDA() = default;

//	template <Distribution Select>
//	auto compare(Id id1, Id id2) const->typename Map2Cmp<Select>::type

public:
	// 以下ユーザインタフェース

	/**
	\brief
		@~japanese ファクトリ関数 (デフォルト設定で使用する場合)	\n
		@~english factory function (construct model with default settings)	\n

	\details
		@~japanese
		ハイパーパラメータは[α = 50/num_topics, β = 0.01, γ = 0.5]に設定．\n
		ハイパーパラメータに関する詳細は \ref g_hparam_setting を参照．

		\param num_topics トピック数（潜在因子の次元数）
		\param input_data 文書データ（DocumentLoader および その派生クラスにより作成）
		\param resume 前回の学習途中のパラメータから学習を再開するか

		@~english
		Hyper-parameters are set as default [α = 50/num_topics, β = 0.01, γ = 0.5]. \n
		See \ref g_hparam_setting about details of hyper-parameters.

		\param num_topics number of topics（dimension of latent factor）
		\param input_data documents（instance of DocumentLoader or its derived class）
		\param resume whether reload previous trained parameters or not
	*/
	static TwitterLDAPtr makeInstance(
		uint num_topics,
		DocumentSetPtr input_data,
		bool resume = false
	){
		return DocumentType::Tweet == input_data->getDocumentType()
			? TwitterLDAPtr(new TwitterLDA(
				resume,
				num_topics,
				input_data,
				Nothing<VectorK<double>>(),
				Nothing<VectorV<double>>(),
				Nothing<VectorB<double>>()
			))
			: nullptr;
	}
	
	
	/**
	\brief
		@~japanese ファクトリ関数 (α, β をsymmetricに設定して使用する場合)	\n
		@~english factory function (construct model with symmetric hyper-parameters)	\n

	\details
		@~japanese
		ハイパーパラメータ（ベクトル）の各要素をすべて同じ値に設定．\n
		デフォルト値を使う場合は sig::nothing または sig::Nothing<double>() を引数に指定．\n
		ハイパーパラメータに関する詳細は \ref sigtm_guide を参照．
		
		\param num_topics トピック数（潜在因子の次元数）
		\param input_data 文書データ（DocumentLoader および その派生クラスにより作成）
		\param alpha ディリクレ分布ハイパーパラメータα
		\param beta ディリクレ分布ハイパーパラメータβ
		\param gamma ディリクレ分布ハイパーパラメータγ
		\param resume 前回の学習途中のパラメータから学習を再開するか

		@~english
		In hyper-parameter vector, each element is set as the same value.	\n
		If want to set as default, pass either sig::nothing or sig::Nothing<double>() to the corresponding argument.	\n
		See \ref g_hparam_setting about details of hyper-parameters.

		\param num_topics number of topics（dimension of latent factor）
		\param input_data documents（instance of DocumentLoader or its derived class）
		\param alpha　hyper-parameter α
		\param beta hyper-parameter β
		\param gamma hyper-parameter γ
		\param resume whether reload previous trained parameters or not
	*/
	static TwitterLDAPtr makeInstance(
		uint num_topics,
		DocumentSetPtr input_data,
		Maybe<double> alpha,
		Maybe<double> beta = Nothing<double>(),
		Maybe<double> gamma = Nothing<double>(),
		bool resume = false
	){
		return DocumentType::Tweet == input_data->getDocumentType()
			? TwitterLDAPtr(new TwitterLDA(
				resume,
				num_topics,
				input_data,
				isJust(alpha) ? Just(VectorK<double>(num_topics, fromJust(alpha))) : Nothing<VectorK<double>>(),
				isJust(beta) ? Just(VectorV<double>(input_data->getWordNum(), fromJust(beta))) : Nothing<VectorV<double>>(),
				isJust(gamma) ? Just(VectorB<double>{fromJust(gamma), 1 - fromJust(gamma)}) : Nothing<VectorB<double>>()
			))
			: nullptr;
	}
	
	
	/** 
	\brief
		@~japanese ファクトリ関数 (α, β をunsymmetricに設定して使用する場合)	\n
		@~english factory function (construct model with unsymmetric hyper-parameters)	\n

	\details
		@~japanese
		ハイパーパラメータのベクトルをすべて任意の値に設定．	\n
		デフォルト値を使う場合は sig::nothing または sig::Nothing<std::vector<double>>() を引数に指定．\n
		ハイパーパラメータに関する詳細は \ref sigtm_guide を参照．

		\param num_topics トピック数（潜在因子の次元数）
		\param input_data 文書データ（DocumentLoader および その派生クラスにより作成）
		\param alpha ディリクレ分布ハイパーパラメータα
		\param beta ディリクレ分布ハイパーパラメータβ
		\param gamma ディリクレ分布ハイパーパラメータγ
		\param resume 前回の学習途中のパラメータから学習を再開するか

		@~english
		In hyper-parameter vector, each element is set as arbitrary value.	\n
		If want to set as default, pass either sig::nothing or sig::Nothing<double>() to the corresponding argument.	\n
		See \ref g_hparam_setting about details of hyper-parameters.

		\param num_topics number of topics（dimension of latent factor）
		\param input_data documents（instance of DocumentLoader or its derived class）
		\param alpha　hyper-parameter α
		\param beta hyper-parameter β
		\param gamma hyper-parameter γ
		\param resume whether reload previous trained parameters or not
	*/
	static TwitterLDAPtr makeInstance(
		uint num_topics,
		DocumentSetPtr input_data,
		Maybe<VectorK<double>> alpha,
		Maybe<VectorV<double>> beta = Nothing<VectorV<double>>(),
		Maybe<VectorB<double>> gamma = Nothing<VectorB<double>>(),
		bool resume = false
	){
		return DocumentType::Tweet == input_data->getDocumentType()
			? TwitterLDAPtr(new TwitterLDA(
				resume,
				num_topics,
				input_data,
				alpha,
				beta,
				gamma
			))
			: nullptr;
	}


	/** 
	\brief
		@~japanese モデルの学習を行う
		@~english Train model
	*/
	void train(uint num_iteration){ train(num_iteration, null_twlda_callback); }
	
	/** 
	\brief
		@~japanese モデルの学習を行う
		@~english Train model
	*/
	void train(uint num_iteration, std::function<void(TwitterLDA const*)> callback);


	/**
	\brief 
		@~japanese コンソールに出力
		@~english Output to console
	*/
	void print(Distribution target) const{ save(target, SIG_TO_FPSTR("")); }

	/**
	\brief 
		@~japanese ファイルに出力
		@~english Output to file
	*/
	void save(Distribution target, FilepassString save_dir, bool detail = false) const;

	/**
	\brief
		@~japanese ユーザが持つトピック比率を取得
		@~english get the topic proportion of each user
	*/
	auto getTheta() const->MatrixDK<double>;

	/**
	\brief
		@~japanese 指定ユーザが持つトピック比率を取得
		@~english get the topic proportion of specified user
	*/
	auto getTheta(UserId u_id) const->VectorK<double>;

	/** 
	\brief
		@~japanese トピックが持つ語彙比率を取得
		@~english Get the word proportion of each topic
	*/
	auto getPhi() const->MatrixKV<double>;

	/**
	\brief
		@~japanese 指定トピックが持つ語彙比率を取得
		@~english Get the word proportion of specified topic
	*/
	auto getPhi(TopicId k_id) const->VectorV<double>;


	auto getPhiBackground() const->VectorV<double>;

	// 全単語におけるtopic[0]とbackground[1]の比率
	auto getY() const->VectorB<double>;

	// 各ユーザにおける単語のtopic[0]とbackground[1]の比率
	auto getEachY() const->MatrixUB<double>;
	auto getEachY(UserId u_id) const->VectorB<double>;

	//トピックを強調する単語スコア
//	auto getTermScore() const->MatrixKV<double> override{ return term_score_; }		// [topic][word]
//	auto getTermScore(TopicId k_id) const->VectorV<double> override{ return term_score_[k_id]; }	// [word]


	/**
	\brief
		@~japanese トピックの代表語彙とそのスコアを取得
		@~english Get the characteristic words and their scores of each topic
	*/
	auto getWordOfTopic(Distribution target, uint num_get_words) const->VectorK<std::vector< std::tuple<std::wstring, double>>>;
	
	/**
	\brief
		@~japanese 指定トピックの代表語彙とそのスコアを取得
		@~english Get the characteristic words and their scores of specific topic
	*/
	auto getWordOfTopic(Distribution target, uint num_get_words, TopicId k_id) const->std::vector< std::tuple<std::wstring, double>>;


	/**
	\brief
		@~japanese 指定ユーザが持つ各tweetのトピック比率を取得
		@~english get the topic proportion of specified user's tweets

	\details
		@~japanese
		\param u_id 取得したいユーザID
		\return result (:: [tweet][topic])

		@~english
		\param u_id user id you want to get
		\return result (:: [tweet][topic])
	*/
	auto getTopicOfTweet(UserId u_id) const->MatrixDK<double>;

	/**
	\brief
		@~japanese 指定tweetが持つトピック比率を取得
		@~english get the topic proportions of specified tweet

	\details
		@~japanese
		\param u_id 取得したいユーザID
		\param d_id 取得したい指定ユーザのtweetID
		\return result (:: [topic])

		@~english
		\param u_id user id you want to get
		\param d_id specified user's tweet id you want to get
		\return result (:: [topic])
	*/
	auto getTopicOfTweet(UserId u_id, DocumentId d_id) const->VectorK<double>;	// [topic]


	// 指定ドキュメントの上位num_get_words個の、語彙とスコアを返す
	auto getWordOfUser(uint num_get_words) const->VectorU<std::vector< std::tuple<std::wstring, double>>>;	
	auto getWordOfUser(uint num_get_words, DocumentId d_id) const->std::vector< std::tuple<std::wstring, double>>;


	/**
	\brief
		@~japanese ユーザ数を取得
		@~english get the number of users
	*/
	uint getUserNum() const{ return U_; }

	auto getTweetNum() const->VectorU<uint>{ return D_; }
	uint getTweetNum(UserId u_id) const{ return D_[u_id]; }

	/**
	\brief
		@~japanese トピック数を取得
		@~english Get the number of topics
	*/
	uint getTopicNum() const{ return K_; }

	/**
	\brief
		@~japanese 語彙数を取得
		@~english Get the number of words (vocabularies)
	*/
	uint getWordNum() const{ return V_; }

	/**
	\brief
		@~japanese ハイパーパラメータαを取得（\ref g_hparam_alpha ）
		@~english Get \ref g_hparam_alpha
	*/
	auto getAlpha() const->VectorK<double>{ return alpha_; }
	
	/**
	\brief
		@~japanese ハイパーパラメータβを取得（\ref g_hparam_beta ）
		@~english Get \ref g_hparam_beta
	*/
	auto getBeta() const->VectorV<double>{ return beta_; }

	// auto getGamma

	/**
	\brief
		@~japanese モデルの対数尤度（\ref g_log_likelihood ）を取得
		@~english Get model \ref g_log_likelihood
	*/
	double getLogLikelihood() const;

	/**
	\brief
		@~japanese モデルの \ref g_perplexity を取得
		@~english Get model \ref g_perplexity
	*/
	double getPerplexity() const{ return std::exp(-getLogLikelihood() / tokens_.size()); }
};

}

namespace std
{
	template <> struct hash<std::tuple<sigtm::UserId, sigtm::DocumentId>>
	{
		size_t operator()(std::tuple<sigtm::UserId, sigtm::DocumentId> const& x) const
		{
			return hash<sigtm::UserId>()(std::get<0>(x)) ^ hash<sigtm::DocumentId>()(std::get<1>(x));
		}
	};
}	//std
#endif
