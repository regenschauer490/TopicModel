/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_TWITTER_LDA_H
#define SIGTM_TWITTER_LDA_H

#include "lda_interface.hpp"
#include "SigUtil/lib/array.hpp"

#include "../helper/input.h"
#if USE_SIGNLP
#include "../helper/input_text.h"
#endif


namespace sigtm
{

template<class T> using VectorB = sig::array<T, 2>;	// for bernoulli parameter
template<class T> using VectorU = std::vector<T>;	// all users
template<class T> using MatrixUD = VectorU<VectorD<T>>;	// user - tweet
template<class T> using MatrixUK = VectorU<VectorK<T>>;	// user - topic
template<class T> using MatrixUDT = VectorU<VectorD<VectorT<T>>>;	// user - tweet - token

class TwitterLDA;
using TwitterLDAPtr = std::shared_ptr<TwitterLDA>;

const std::function<void(TwitterLDA const*)> null_twlda_callback = [](TwitterLDA const*){};


class TwitterLDA : private LDA
{
	InputDataPtr input_data_;
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

	TwitterLDA(bool resume, uint topic_num, InputDataPtr input_data, maybe<VectorK<double>> alpha, maybe<VectorV<double>> beta, maybe<VectorB<double>> gamma) :
		input_data_(input_data), tokens_(input_data->tokens_), U_(input_data->getDocNum()), K_(topic_num), V_(input_data->getWordNum()),
		alpha_(alpha ? sig::fromJust(alpha) : SIG_INIT_VECTOR(double, K, default_alpha_base / K_)), beta_(beta ? sig::fromJust(beta) : SIG_INIT_VECTOR(double, V, default_beta)), gamma_(gamma ? sig::fromJust(gamma) : VectorB<double>{0.5, 0.5}),
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
	/* InputDataで作成した入力データを元にコンストラクト */
	// デフォルト設定で使用する場合
	static TwitterLDAPtr makeInstance(bool resume, uint topic_num, InputDataPtr input_data){
		return DocumentType::Tweet == input_data->doc_type_
			? TwitterLDAPtr(new TwitterLDA(resume, topic_num, input_data, nothing, nothing, nothing))
			: nullptr;
	}
	// alpha, beta をsymmetricに設定する場合
	static TwitterLDAPtr makeInstance(bool resume, uint topic_num, InputDataPtr input_data, double alpha, maybe<double> gamma = nothing, maybe<double> beta = nothing){
		return TwitterLDAPtr(new TwitterLDA(resume, topic_num, input_data, VectorK<double>(topic_num, alpha),
			beta ? sig::Just<VectorV<double>>(VectorV<double>(input_data->getWordNum(), sig::fromJust(beta))) : nothing,
			gamma ? sig::Just<VectorB<double>>(VectorB<double>{sig::fromJust(gamma), 1 - sig::fromJust(gamma)}) : nothing)
		);
	}
	// alpha, beta を多次元で設定する場合
	template <class SamplingMethod = CollapsedGibbsSampling>
	static LDAPtr makeInstance(bool resume, uint topic_num, InputDataPtr input_data, VectorK<double> alpha, maybe<VectorV<double>> beta = nothing){
		return LDAPtr(new LDA_Gibbs(SamplingMethod(), resume, topic_num, input_data, alpha, beta));
	}

	/* モデルの学習を行う */
	// iteration_num: 学習の反復回数(ギブスサンプリングによる全変数の更新を1反復とする)
	void train(uint iteration_num){ train(iteration_num, null_twlda_callback); }

	// call_back: 毎回の反復終了時に行う処理
	void train(uint iteration_num, std::function<void(TwitterLDA const*)> callback);


	// 確率分布同士の類似度を測る(メソッドチェーンな感じに使用)
	// Select: TwitterLDA::Distributionから選択
	// id1,id2: 類似度を測る対象のindex
	// return -> 比較関数の選択(関数オブジェクト)
//	template <Distribution Select>
//	auto compare(Id id1, Id id2) const->typename Map2Cmp<Select>::type

	// コンソールに出力
	void print(Distribution target) const{ save(target, L""); }

	// ファイルに出力
	// save_folder: 保存先のフォルダのパス
	// detail: 詳細なデータも全て出力するか
	void save(Distribution target, FilepassString save_folder, bool detail = false) const;

	// ユーザのトピック分布
	auto getTheta() const->MatrixUK<double>{ return LDA::getTheta(); }	// [user][topic]
	auto getTheta(UserId u_id) const->VectorK<double>;		// [topic]

	// tweetのトピック分布
	auto getTopicOfTweet(UserId u_id) const->MatrixDK<double>;		// [tweet][topic]
	auto getTopicOfTweet(UserId u_id, DocumentId d_id) const->VectorK<double>;	// [topic]

	//トピックの単語分布
	auto getPhi() const->MatrixKV<double>{ return LDA::getPhi(); }	// [topic][word]
	auto getPhi(TopicId k_id) const->VectorV<double>;		// [word]

	// backgroundの単語分布
	auto getPhiBackground() const->VectorV<double>;	// [word]

	//トピックを強調する単語スコア
//	auto getTermScore() const->MatrixKV<double> override{ return term_score_; }		// [topic][word]
//	auto getTermScore(TopicId t_id) const->VectorV<double> override{ return term_score_[t_id]; }	// [word]

	// 指定トピックの上位return_word_num個の、語彙とスコアを返す
	auto getWordOfTopic(Distribution target, uint return_word_num) const->VectorK<std::vector< std::tuple<std::wstring, double>>>;		// [topic][ranking]<vocab, score>	
	auto getWordOfTopic(Distribution target, uint return_word_num, TopicId k_id) const->std::vector< std::tuple<std::wstring, double>>;	// [ranking]<vocab, score>

	// 指定ドキュメントの上位return_word_num個の、語彙とスコアを返す
	auto getWordOfUser(uint return_word_num) const->VectorU<std::vector< std::tuple<std::wstring, double>>>;		// [user][ranking]<vocab, score>
	auto getWordOfUser(uint return_word_num, DocumentId d_id) const->std::vector< std::tuple<std::wstring, double>>;	//[ranking]<vocab, score>

	uint getUserNum() const{ return U_; }
	auto getTweetNum() const->VectorU<uint>{ return D_; }
	uint getTweetNum(UserId u_id) const{ return D_[u_id]; }
	uint getTopicNum() const{ return K_; }
	uint getWordNum() const{ return V_; }

	// get hyper-parameter of topic distribution
	auto getAlpha() const->VectorK<double> override{ return alpha_; }
	// get hyper-parameter of word distribution
	auto getBeta() const->VectorV<double> override{ return beta_; }

	// 
	double getLogLikelihood() const override{ return calcLogLikelihood(tokens_); }

	double getPerplexity() const override{ return std::exp(-getLogLikelihood() / tokens_.size()); }
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