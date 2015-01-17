/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_CTR_HPP
#define SIGTM_CTR_HPP

#define _SCL_SECURE_NO_WARNINGS
#define NDEBUG

#include "../helper/data_format.hpp"
#include "../helper/rating_matrix.hpp"
#include "lda_common_module.hpp"

#if SIG_USE_EIGEN
#include "Eigen/Core"
#else
#include "SigUtil/lib/calculation/ublas.hpp"
#endif

namespace sigtm
{

template <class T>
using MatrixUI = std::vector<std::vector<T>>;

#if SIG_USE_EIGEN

using VectorK_ = EigenVector;		// topic
using VectorV_ = EigenVector;		// word
using MatrixIK_ = EigenMatrix;		// item - topic(factor)
using MatrixUK_ = EigenMatrix;		// user - topic(factor)
using MatrixKK_ = EigenMatrix;
using MatrixKV_ = EigenMatrix;
using MatrixTK_ = EigenMatrix;

#else
using VectorK_ = sig::vector_u<double>;		// topic
using VectorV_ = sig::vector_u<double>;		// word
using MatrixIK_ = sig::matrix_u<double>;	// item - topic(factor)
using MatrixUK_ = sig::matrix_u<double>;	// user - topic(factor)
using MatrixKK_ = sig::matrix_u<double>;
using MatrixKV_ = sig::matrix_u<double>;
using MatrixTK_ = sig::matrix_u<double>;
#endif

struct CtrHyperparameter : boost::noncopyable
{
	std::vector<VectorK<double>> theta_;
	VectorK<VectorV<double>> beta_;
	uint topic_num_;
	double a_;				// positive update weight in U,V
	double b_;				// negative update weight in U,V (b < a)
	double lambda_u_;
	double lambda_v_;
	double learning_rate_;	// stochastic version for large datasets. Stochastic learning will be called when > 0
	bool theta_opt_;
	bool enable_recommend_cache_;

private:
	CtrHyperparameter(uint topic_num, bool optimize_theta, bool enable_recommend_cache)
	{
		topic_num_ = topic_num;
		a_ = 1;
		b_ = 0.01;
		lambda_u_ = 0.01;
		lambda_v_ = 100;
		learning_rate_ = -1;
		theta_opt_ = optimize_theta;
		enable_recommend_cache_ = enable_recommend_cache;
	}

public:
	static auto makeInstance(uint topic_num, bool optimize_theta, bool enable_recommend_cache) ->std::shared_ptr<CtrHyperparameter>{
		return std::shared_ptr<CtrHyperparameter>(new CtrHyperparameter(topic_num, optimize_theta, enable_recommend_cache));
	}

	void setTheta(std::vector<VectorK<double>> const& init){
		theta_ = init;
	}
	void setBeta(VectorK<VectorV<double>> const& init){
		beta_ = init;
	}
};

using CTRHyperParamPtr = std::shared_ptr<CtrHyperparameter>;

//template <class RatingValueType>
class CTR : private impl::LDA_Module
{
public:
	using RatingValueType = int;
	using RatingPtr_ = RatingPtr<RatingValueType>;
	using EstValueType = std::pair<Id, double>;

private:
	using RatingIter = SparseBooleanMatrix::const_iterator;
	using RatingContainer = SparseBooleanMatrix::const_rating_range;
	
	const int model_id_;
	const CTRHyperParamPtr hparam_;
	const DocumentSetPtr input_data_;
	const RatingMatrixPtr<RatingValueType> ratings_;
	const TokenList& tokens_;

	const VectorI<std::vector<TokenId>> item_tokens_;	// tokens in each item(document)

	const RatingContainer user_ratings_;
	const RatingContainer item_ratings_;

	const uint T_;		// number of tokens
	const uint K_;		// number of topics(factor)
	const uint V_;		// number of words	
	const uint U_;		// number of users
	const uint I_;		// number of items

	MatrixKV_ beta_;	// word distribution of topic
	MatrixIK_ theta_;
	MatrixUK_ user_factor_;
	MatrixIK_ item_factor_;

	mutable Maybe<MatrixUI<Maybe<double>>> estimate_ratings_;
	mutable Maybe<MatrixKV<double>> term_score_;	// word score of emphasizing each topic

	double likelihood_;
	const double conv_epsilon_ = 1e-4;

	// temporary
	VectorK_ gamma_;
	MatrixKV_ log_beta_;
	MatrixKV_ word_ss_;
	MatrixTK_ phi_;

private:
	void init();

	void printUFactor() const;
	void printIFactor() const;

	void saveTmp() const;
	void save() const;
	void load();

	double docInference(ItemId id, bool update_word_ss);

	void updateU();
	void updateV();
	void updateBeta();

	auto recommend_impl(Id id, bool for_user, bool ignore_train_set = true) const->std::vector<std::pair<Id, double>>;
	
private:
	CTR(CTRHyperParamPtr hparam, DocumentSetPtr docs, RatingMatrixPtr<RatingValueType> ratings, int model_id)
	: model_id_(model_id), hparam_(hparam), input_data_(docs), ratings_(ratings), tokens_(docs->tokens_), item_tokens_(docs->getDevidedDocument()),
		user_ratings_(ratings->getUsers()), item_ratings_(ratings->getItems()), T_(docs->getTokenNum()), K_(hparam->topic_num_), V_(docs->getWordNum()),
		U_(ratings->userSize()), I_(ratings->itemSize()), beta_(K_, V_), theta_(I_, K_), user_factor_(U_, K_), item_factor_(I_, K_),
		estimate_ratings_(nothing), likelihood_(-std::exp(50)),	gamma_(K_), log_beta_(K_, V_), word_ss_(K_, V_), phi_(T_, K_)
	{
		init();
	}
	CTR(CTRHyperParamPtr hparam, DocumentSetPtr docs, RatingMatrixPtr<RatingValueType> ratings)
	: CTR(hparam, docs, ratings, -1) {}
	
public:	
	static auto makeInstance(CTRHyperParamPtr hparam, DocumentSetPtr docs, RatingMatrixPtr<RatingValueType> ratings) ->std::shared_ptr<CTR>
	{
		return std::shared_ptr<CTR>(new CTR(hparam, docs, ratings));
	}
	static auto makeInstance(CTRHyperParamPtr hparam, DocumentSetPtr docs, RatingMatrixPtr<RatingValueType> ratings, uint model_id) ->std::shared_ptr<CTR>
	{
		return std::shared_ptr<CTR>(new CTR(hparam, docs, ratings, model_id));
	}

	void train(uint max_iter, uint min_iter, uint save_lag);

	// return recommended item(for user) or user(for item) list (descending by estimated rating value)
	auto recommend(Id id, bool for_user, sig::Maybe<uint> top_n, sig::Maybe<double> threshold) const->std::vector<std::pair<Id, double>>;

	double estimate(UserId u_id, ItemId i_id) const;
	

	//ドキュメントのトピック比率
	auto getTheta() const->MatrixIK_;		// [doc][topic]
	auto getTheta(ItemId i_id) const->VectorK_;	// [topic]
	
	//トピックの単語比率
	auto getPhi() const->MatrixKV_;		// [topic][word]
	auto getPhi(TopicId k_id) const->VectorV_;	// [word]
	
	//トピックを強調する単語スコア
	auto getTermScore() const->MatrixKV<double>;		// [topic][word]
	auto getTermScore(TopicId t_id) const->VectorV<double>;	// [word]

	// 指定トピックの上位return_word_num個の、語彙とスコアを返す
	auto getWordOfTopic(TopicId k_id, uint return_word_num, bool calc_term_score = true) const->std::vector< std::tuple<std::wstring, double>>;	// [ranking]<vocab, score>

	uint getUserNum() const { return U_; }
	uint getItemNum() const { return I_; }
	uint getTopicNum() const { return K_; }
	uint getWordNum() const { return V_; }
	uint getUserRatingNum(uint user_id) const { return user_ratings_[user_id].size(); }
	uint getItemRatingNum(uint item_id) const { return item_ratings_[item_id].size(); }

	void debug_set_u(std::vector<std::vector<double>> const& v) {
		for (uint i = 0; i < v.size(); ++i) {
			for (uint j = 0; j < v[i].size(); ++j) user_factor_(i, j) = v[i][j];
		}
	}
	void debug_set_v(std::vector<std::vector<double>> const& v){
		for (uint i = 0; i < v.size(); ++i) {
			for (uint j = 0; j < v[i].size(); ++j)item_factor_(i, j) = v[i][j];
		}
	}
};

using CTRPtr = std::shared_ptr<CTR>;
}	// sigtm

#endif