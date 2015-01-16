/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_LDA_GIBBS_H
#define SIGTM_LDA_GIBBS_H

#include "lda_common_module.hpp"

namespace sigtm
{
/* Latent Dirichlet Allocation (estimate by Gibbs Sampling or Collapsed Gibbs Sampling) */
class LDA_Gibbs : public LDA, private impl::LDA_Module
{
	DocumentSetPtr input_data_;
	TokenList const& tokens_;
	
	const uint D_;		// number of documents
	const uint K_;		// number of topics
	const uint V_;		// number of words

	VectorK<double> alpha_;			// dirichlet hyper parameter of theta
	VectorV<double> beta_;			// dirichlet hyper parameter of phi
	
	MatrixVK<uint> word_ct_;		// token count of each topic in each word
	MatrixDK<uint> doc_ct_;			// token count of each topic in each document
	VectorK<uint> topic_ct_;		// token count of each topic

	VectorT<uint> z_;				// topic assigned to each token
			
	double alpha_sum_;
	double beta_sum_;
	VectorK<double> tmp_p_;
	MatrixKV<double> term_score_;	// word score of emphasizing each topic
	uint total_iter_ct_;
	
	const std::function<double(LDA_Gibbs const* obj, DocumentId d, WordId v, TopicId k)> sampling_;
	sig::SimpleRandom<uint> rand_ui_;
	sig::SimpleRandom<double> rand_d_;

public:
	struct GibbsSampling
	{
		double operator()(LDA_Gibbs const* obj, DocumentId d, WordId v, TopicId k)
		{
			static int pre_k = -1;
			static double dk_sum = 0;
			double const& alpha = obj->alpha_[k];
			double const& beta = obj->beta_[v];
			
			if(k != pre_k) dk_sum = static_cast<double>(sig::sum(obj->doc_ct_[d]));
			pre_k = k;
			return ((obj->doc_ct_[d][k] + alpha) / (dk_sum + obj->alpha_sum_)) * ((obj->word_ct_[v][k] + beta) / (obj->topic_ct_[k] + obj->beta_sum_));
		}
	};

	struct CollapsedGibbsSampling
	{
		double operator()(LDA_Gibbs const* obj, DocumentId d, WordId v, TopicId k){
			double const& beta = obj->beta_[v];
			return (obj->doc_ct_[d][k] + obj->alpha_[k]) * (obj->word_ct_[v][k] + beta) / (obj->topic_ct_[k] + obj->beta_sum_);
		}
	};

private:
	LDA_Gibbs() = delete;
	LDA_Gibbs(LDA_Gibbs const&) = delete;

	template <class SamplingMethod>
	LDA_Gibbs(SamplingMethod sm,bool resume,  uint topic_num, DocumentSetPtr input_data, Maybe<VectorK<double>> alpha, Maybe<VectorV<double>> beta) :
		input_data_(input_data), tokens_(input_data->tokens_), D_(input_data->getDocNum()), K_(topic_num), V_(input_data->getWordNum()),
		alpha_(alpha ? sig::fromJust(alpha) : SIG_INIT_VECTOR(double, K, default_alpha_base / K_)), beta_(beta ? sig::fromJust(beta) : SIG_INIT_VECTOR(double, V, default_beta)),
		word_ct_(SIG_INIT_MATRIX(uint, V, K, 0)), doc_ct_(SIG_INIT_MATRIX(uint, D, K, 0)), topic_ct_(SIG_INIT_VECTOR(uint, K, 0)),
		z_(tokens_.size(), 0), tmp_p_(SIG_INIT_VECTOR(double, K, 0)), term_score_(SIG_INIT_MATRIX(double, K, V, 0)), total_iter_ct_(0),
		sampling_(SamplingMethod()), rand_ui_(0, K_ - 1, FixedRandom), rand_d_(0.0, 1.0, FixedRandom)
	{
		init(resume);
	}

	void init(bool resume);
	void saveResumeData() const;
	void update(Token const& t);

public:
	~LDA_Gibbs(){}

	DynamicType getDynamicType() const override{ return DynamicType::GIBBS; }

	/* DocumentSetのデータからコンストラクト */
	// デフォルト設定で使用する場合
	template <class SamplingMethod = CollapsedGibbsSampling>
	static LDAPtr makeInstance(bool resume, uint topic_num, DocumentSetPtr input_data){
		return LDAPtr(new LDA_Gibbs(SamplingMethod(), resume, topic_num, input_data, nothing, nothing));
	}
	// alpha, beta をsymmetricに設定する場合
	template <class SamplingMethod = CollapsedGibbsSampling>
	static LDAPtr makeInstance(bool resume, uint topic_num, DocumentSetPtr input_data, double alpha, Maybe<double> beta = nothing){
		return LDAPtr(new LDA_Gibbs(SamplingMethod(), resume, topic_num, input_data, VectorK<double>(topic_num, alpha), beta ? sig::Just<VectorV<double>>(VectorV<double>(input_data->getWordNum(), sig::fromJust(beta))) : nothing));
	}
	// alpha, beta を多次元で設定する場合
	template <class SamplingMethod = CollapsedGibbsSampling>
	static LDAPtr makeInstance(bool resume, uint topic_num, DocumentSetPtr input_data, VectorK<double> alpha, Maybe<VectorV<double>> beta = nothing){
		return LDAPtr(new LDA_Gibbs(SamplingMethod(), resume, topic_num, input_data, alpha, beta));
	}
	
	/* モデルの学習を行う */
	// iteration_num: 学習の反復回数(ギブスサンプリングによる全変数の更新を1反復とする)
	void train(uint iteration_num) override{ train(iteration_num, null_lda_callback);  }

	// call_back: 毎回の反復終了時に行う処理
	void train(uint iteration_num, std::function<void(LDA const*)> callback) override;


	// 確率分布同士の類似度を測る(メソッドチェーンな感じに使用)
	// Select: LDA::Distributionから選択
	// id1,id2: 類似度を測る対象のindex
	// return -> 比較関数の選択(関数オブジェクト)
	template <Distribution Select>
	auto compare(Id id1, Id id2) const->typename Map2Cmp<Select>::type{	return compareDefault<Select>(id1, id2, D_, K_); }

	// コンソールに出力
	void print(Distribution target) const override{ save(target, L""); }

	// ファイルに出力
	void save(Distribution target, FilepassString save_folder, bool detail = false) const override;

	//ドキュメントのトピック比率
	using LDA::getTheta;		// [doc][topic]
	auto getTheta(DocumentId d_id) const->VectorK<double> override;	// [topic]

	//トピックの単語比率
	using LDA::getPhi;		// [topic][word] //std::bind(std::mem_fn<VectorV<double>, LDA_Gibbs, TopicId>(LDA_Gibbs::getPhi), *this,)
	auto getPhi(TopicId k_id) const->VectorV<double> override;	// [word]

	//トピックを強調する単語スコア
	auto getTermScore() const->MatrixKV<double> override{ return term_score_; }		// [topic][word]
	auto getTermScore(TopicId t_id) const->VectorV<double> override{ return term_score_[t_id]; }	// [word]

	// 指定トピックの上位return_word_num個の、語彙とスコアを返す
	using LDA::getWordOfTopic;		// [topic][ranking]<vocab, score>
	auto getWordOfTopic(Distribution target, uint return_word_num, TopicId k_id) const->std::vector< std::tuple<std::wstring, double>> override;	// [ranking]<vocab, score>

	// 指定ドキュメントの上位return_word_num個の、語彙とスコアを返す
	using LDA::getWordOfDocument;		// [doc][ranking]<vocab, score>	
	auto getWordOfDocument(uint return_word_num, DocumentId d_id) const->std::vector< std::tuple<std::wstring, double>> override;	//[ranking]<vocab, score>

	uint getDocumentNum() const override{ return D_; }
	uint getTopicNum() const override{ return K_; }
	uint getWordNum() const override{ return V_; }

	// get hyper-parameter of topic distribution
	auto getAlpha() const->VectorK<double> override{ return alpha_; }
	// get hyper-parameter of word distribution
	auto getBeta() const->VectorV<double> override{ return beta_; }


	double getLogLikelihood() const override{ return calcLogLikelihood(tokens_, getTheta(), getPhi()); }

	double getPerplexity() const override{ return std::exp(-getLogLikelihood() / tokens_.size()); }
};

}
#endif
