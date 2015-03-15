/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_LDA_GIBBS_H
#define SIGTM_LDA_GIBBS_H

#include "common/lda_module.hpp"

namespace sigtm
{
/// Latent Dirichlet Allocation (estimate by Gibbs Sampling or Collapsed Gibbs Sampling)
/**
*/
class LDA_Gibbs final : public LDA, private impl::LDA_Module
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
			using sig::operator<<=;

			static auto pre_k = Nothing<uint>();
			static double dk_sum = 0;
			double const& alpha = obj->alpha_[k];
			double const& beta = obj->beta_[v];
			
			if(k != *pre_k) dk_sum = static_cast<double>(sig::sum(obj->doc_ct_[d]));
			pre_k <<= k;
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
	LDA_Gibbs(SamplingMethod sm, bool resume, uint num_topics, DocumentSetPtr input_data, Maybe<VectorK<double>> alpha, Maybe<VectorV<double>> beta) :
		input_data_(input_data), tokens_(input_data->tokens_), D_(input_data->getDocNum()), K_(num_topics), V_(input_data->getWordNum()),
		alpha_(isJust(alpha) ? fromJust(alpha) : SIG_INIT_VECTOR(double, K, default_alpha_base / K_)), beta_(isJust(beta) ? fromJust(beta) : SIG_INIT_VECTOR(double, V, default_beta)),
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
	~LDA_Gibbs() = default;

	DynamicType getDynamicType() const override{ return DynamicType::GIBBS; }
	
	template <Distribution Select>
	auto compare(Id id1, Id id2) const->typename Map2Cmp<Select>::type { return compareDefault<Select>(id1, id2, D_, K_); }

public:
	// 以下ユーザインタフェース

	/**
	\brief
		@~japanese ファクトリ関数 (デフォルト設定で使用する場合)	\n
		@~english factory function (construct model with default settings)	\n

	\details
		@~japanese
		ハイパーパラメータは[α = 50/num_topics, β = 0.01]に設定．\n
		ハイパーパラメータに関する詳細は \ref g_hparam_setting を参照．

		\tparam SamplingMethod GibbsSampling か CollapsedGibbsSampling かを選択

		\param num_topics トピック数（潜在因子の次元数）
		\param input_data 文書データ（DocumentLoader および その派生クラスにより作成）
		\param resume 前回の学習途中のパラメータから学習を再開するか

		@~english
		Hyper-parameters are set as default [α = 50/num_topics, β = 0.01]. \n
		See \ref g_hparam_setting about details of hyper-parameters.

		\tparam SamplingMethod select GibbsSampling or CollapsedGibbsSampling

		\param num_topics number of topics（dimension of latent factor）
		\param input_data documents（instance of DocumentLoader or its derived class）
		\param resume whether reload previous trained parameters or not
	*/
	template <class SamplingMethod = CollapsedGibbsSampling>
	static LDAPtr makeInstance(
		uint num_topics,
		DocumentSetPtr input_data,
		bool resume = false
	){
		return LDAPtr(new LDA_Gibbs(
			SamplingMethod(),
			resume,
			num_topics,
			input_data,
			Nothing<VectorK<double>>(),
			Nothing<VectorV<double>>()
		));
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

		\tparam SamplingMethod GibbsSampling か CollapsedGibbsSampling かを選択

		\param num_topics トピック数（潜在因子の次元数）
		\param input_data 文書データ（DocumentLoader および その派生クラスにより作成）
		\param alpha ディリクレ分布ハイパーパラメータα
		\param beta ディリクレ分布ハイパーパラメータβ
		\param resume 前回の学習途中のパラメータから学習を再開するか

		@~english
		In hyper-parameter vector, each element is set as the same value.	\n
		If want to set as default, pass either sig::nothing or sig::Nothing<double>() to the corresponding argument.	\n
		See \ref g_hparam_setting about details of hyper-parameters.

		\tparam SamplingMethod select GibbsSampling or CollapsedGibbsSampling

		\param num_topics number of topics（dimension of latent factor）
		\param input_data documents（instance of DocumentLoader or its derived class）
		\param alpha　hyper-parameter α
		\param beta hyper-parameter β
		\param resume whether reload previous trained parameters or not
	*/
	template <class SamplingMethod = CollapsedGibbsSampling>
	static LDAPtr makeInstance(
		uint num_topics,
		DocumentSetPtr input_data,
		Maybe<double> alpha,
		Maybe<double> beta = :Nothing<double>(),
		bool resume = false
	){
		return LDAPtr(new LDA_Gibbs(
			SamplingMethod(),
			resume,
			num_topics,
			input_data,
			isJust(alpha) ? Just(VectorK<double>(num_topics, fromJust(alpha))) : Nothing<VectorK<double>>(),
			isJust(beta) ? Just(VectorV<double>(input_data->getWordNum(), fromJust(beta))) : Nothing<VectorV<double>>()
		));
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

		\tparam SamplingMethod GibbsSampling か CollapsedGibbsSampling かを選択

		\param num_topics トピック数（潜在因子の次元数）
		\param input_data 文書データ（DocumentLoader および その派生クラスにより作成）
		\param alpha ディリクレ分布ハイパーパラメータα
		\param beta ディリクレ分布ハイパーパラメータβ
		\param resume 前回の学習途中のパラメータから学習を再開するか

		@~english
		In hyper-parameter vector, each element is set as arbitrary value.	\n
		If want to set as default, pass either sig::nothing or sig::Nothing<double>() to the corresponding argument.	\n
		See \ref g_hparam_setting about details of hyper-parameters.

		\tparam SamplingMethod select GibbsSampling or CollapsedGibbsSampling

		\param num_topics number of topics（dimension of latent factor）
		\param input_data documents（instance of DocumentLoader or its derived class）
		\param alpha　hyper-parameter α
		\param beta hyper-parameter β
		\param resume whether reload previous trained parameters or not
	*/
	template <class SamplingMethod = CollapsedGibbsSampling>
	static LDAPtr makeInstance(
		uint num_topics,
		DocumentSetPtr input_data,
		Maybe<VectorK<double>> alpha,
		Maybe<VectorV<double>> beta = Nothing<VectorV<double>>(),
		bool resume = false
	){
		return LDAPtr(new LDA_Gibbs(
			SamplingMethod(),
			resume, 
			num_topics,
			input_data,
			alpha,
			beta
		));
	}
	

	/** 
	\brief
		@~japanese モデルの学習を行う
		@~english Train model
	*/
	void train(uint num_iteration) override{ train(num_iteration, null_lda_callback);  }

	/** 
	\brief
		@~japanese モデルの学習を行う
		@~english Train model
	*/
	void train(uint num_iteration, std::function<void(LDA const*)> callback) override;

	
	/**
	\brief 
		@~japanese コンソールに出力
		@~english Output to console
	*/
	void print(Distribution target) const override{ save(target, SIG_TO_FPSTR("")); }

	/**
	\brief 
		@~japanese ファイルに出力
		@~english Output to file
	*/
	void save(Distribution target, FilepassString save_dir, bool detail = false) const override;


	/**
	\brief
		@~japanese 文書が持つトピック比率を取得
		@~english Get the topic proportion of each document
	*/
	auto getTheta() const->MatrixDK<double> override{ return LDA::getTheta(); }

	/**
	\brief
		@~japanese 指定文書が持つトピック比率を取得
		@~english Get the topic proportion of specified document
	*/
	auto getTheta(DocumentId d_id) const->VectorK<double> override;


	/** 
	\brief
		@~japanese トピックが持つ語彙比率を取得
		@~english Get the word proportion of each topic
	*/
	auto getPhi() const->MatrixKV<double> override{ return LDA::getPhi(); }

	/**
	\brief
		@~japanese 指定トピックが持つ語彙比率を取得
		@~english Get the word proportion of specified topic
	*/
	auto getPhi(TopicId k_id) const->VectorV<double> override;

	/**
	\brief
		@~japanese トピックを強調する語彙スコアを取得
		@~english Get the word-scores of each topic
	*/
	auto getTermScore() const->MatrixKV<double> override{ return term_score_; }

	/**
	\brief
		@~japanese 指定トピックを強調する語彙スコアを取得
		@~english Get the word-scores of specified topic
	*/
	auto getTermScore(TopicId k_id) const->VectorV<double> override{ return term_score_[k_id]; }
	
	/**
	\brief
		@~japanese トピックの代表語彙とそのスコアを取得
		@~english Get the characteristic words and their scores of each topic
	*/
	auto getWordOfTopic(Distribution target, uint num_get_words) const->VectorK< std::vector< std::tuple<std::wstring, double>>> override{ return LDA::getWordOfTopic(target, num_get_words); }

	/**
	\brief
		@~japanese 指定トピックの代表語彙とそのスコアを取得
		@~english Get the characteristic words and their scores of specific topic
	*/
	auto getWordOfTopic(Distribution target, uint num_get_words, TopicId k_id) const->std::vector< std::tuple<std::wstring, double>> override;

	/**
	\brief
		@~japanese 文書の代表語彙とそのスコアを取得
		@~english Get the characteristic words and their scores of each document
	*/
	auto getWordOfDocument(Distribution target, uint num_get_words) const->VectorD< std::vector< std::tuple<std::wstring, double> > > override{ return LDA::getWordOfDocument(target, num_get_words); }

	/**
	\brief
		@~japanese 指定文書の代表語彙とそのスコアを取得
		@~english Get the characteristic words and their scores of specified document
	*/
	auto getWordOfDocument(Distribution target, uint num_get_words, DocumentId d_id) const->std::vector< std::tuple<std::wstring, double>> override;	//[ranking]<vocab, score>

	/**
	\brief
		@~japanese 文書数を取得
		@~english Get the number of documents
	*/
	uint getDocumentNum() const override{ return D_; }

	/**
	\brief
		@~japanese トピック数を取得
		@~english Get the number of topics
	*/
	uint getTopicNum() const override{ return K_; }

	/**
	\brief
		@~japanese 語彙数を取得
		@~english Get the number of words (vocabularies)
	*/
	uint getWordNum() const override{ return V_; }

	/**
	\brief
		@~japanese ハイパーパラメータαを取得（\ref g_hparam_alpha ）
		@~english Get \ref g_hparam_alpha
	*/
	auto getAlpha() const->VectorK<double> override{ return alpha_; }
	
	/**
	\brief
		@~japanese ハイパーパラメータβを取得（\ref g_hparam_beta ）
		@~english Get \ref g_hparam_beta
	*/
	auto getBeta() const->VectorV<double> override{ return beta_; }


	/**
	\brief
		@~japanese モデルの対数尤度（\ref g_log_likelihood ）を取得
		@~english Get model \ref g_log_likelihood
	*/
	double getLogLikelihood() const override{ return calcLogLikelihood(tokens_, getTheta(), getPhi()); }

	/**
	\brief
		@~japanese モデルの \ref g_perplexity を取得
		@~english Get model \ref g_perplexity
	*/
	double getPerplexity() const override{ return std::exp(-getLogLikelihood() / tokens_.size()); }
};

}
#endif
