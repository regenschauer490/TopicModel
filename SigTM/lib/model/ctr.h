
#ifndef SIGTM_CTR_HPP
#define SIGTM_CTR_HPP

#define _SCL_SECURE_NO_WARNINGS

#include "../sigtm.hpp"
#include "../helper/data_format.hpp"
#include "lda_common_module.hpp"
#include "SigUtil/lib/calculation/ublas.hpp"
//#include <boost/numeric/ublas/matrix_sparse.hpp>

namespace sigtm
{

using ItemId = uint;
template<class T> using VectorI = VectorD<T>;			// item
template<class T> using VectorK_ = sig::vector_u<T>;	// topic
template<class T> using MatrixIK = VectorI<VectorK<T>>;	// item - topic(factor)

template<class T> using MatrixIK_ = sig::matrix_u<T>;	// item - topic(factor)
template<class T> using MatrixUK_ = sig::matrix_u<T>;	// user - topic(factor)
template<class T> using MatrixKK_ = sig::matrix_u<T>;
template<class T> using MatrixKV_ = sig::matrix_u<T>;
template<class T> using MatrixTK_ = sig::matrix_u<T>;

struct CtrHyperparameter
{
	double a_;				// positive item weight
	double b_;				// negative item weight(b < a)
	double lambda_u_;
	double lambda_v_;
	double learning_rate_;	// stochastic version for large datasets. Stochastic learning will be called when > 0
	double alpha_smooth_;
	double beta_smooth_;
	bool theta_opt_;
	bool lda_regression_;

	CtrHyperparameter(bool optimize_theta, bool run_lda_regression)
	{
		a_ = 1;
		b_ = 0.01;
		lambda_u_ = 0.01;
		lambda_v_ = 100;
		learning_rate_ = -1;
		alpha_smooth_ = 1.0;
		beta_smooth_ =default_beta;
		theta_opt_ = optimize_theta;
		lda_regression_ = run_lda_regression;
	}
	/*
	void set(double a, double b, double lu, double lv, double lr, double as, int to, int lda_r)
	{
		a_ = a;
		b_ = b;
		lambda_u_ = lu;
		lambda_v_ = lv;
		learning_rate_ = lr;
		alpha_smooth_ = as;
		theta_opt_ = to;
		lda_regression_ = lda_r;
	}*/
};


struct Rating : boost::noncopyable
{
	//int value_;
	uint user_id_;
	uint item_id_;

	Rating(uint user_id, uint item_id) : user_id_(user_id), item_id_(item_id){}
};

using RatingPtr = std::shared_ptr<Rating>;

class BooleanMatrix : boost::noncopyable
{
	VectorU<std::vector<RatingPtr>> user_;
	VectorI<std::vector<RatingPtr>> item_;

public:
	BooleanMatrix(std::vector<std::vector<Id>> const& ratings, bool is_user_ratings)
	{
		std::unordered_map<Id, std::vector<RatingPtr>> id_rating_map;
		uint row_id = 0;

		if (is_user_ratings) user_.resize(ratings.size());
		else item_.resize(ratings.size());

		for (auto const& row : ratings){
			for (Id id : row){
				if (is_user_ratings){
					auto rating = std::make_shared<Rating>(row_id, id);
					user_[row_id].push_back(rating);
					id_rating_map[id].push_back(rating);
				}
				else{
					auto rating = std::make_shared<Rating>(id, row_id);
					item_[row_id].push_back(rating);
					id_rating_map[id].push_back(rating);
				}
			}
			++row_id;
		}

		const uint col_size = id_rating_map.size();

		if (is_user_ratings) item_.resize(col_size);
		else user_.resize(col_size);

		for (Id id = 0; id < col_size; ++id){
			if (is_user_ratings){
				item_[id] = std::move(id_rating_map[id]);
			}
			else{
				user_[id] = std::move(id_rating_map[id]);
			}
		}
	}

	auto getUsers() const->VectorU<std::vector<RatingPtr>> const&{ return user_; }
	auto getItems() const->VectorI<std::vector<RatingPtr>> const&{ return item_; }
};


class CTR
{
	const CtrHyperparameter hparam_;
	const DocumentSetPtr input_data_;
	const TokenList& tokens_;

	const VectorI<std::vector<TokenId>> item_token_;	// tokens in each item(document)

	const VectorU<std::vector<RatingPtr>>& user_rating_;
	const VectorI<std::vector<RatingPtr>>& item_rating_;

	const uint T_;		// number of tokens
	const uint K_;		// number of topics(factor)
	const uint V_;		// number of words	
	const uint U_;		// number of users
	const uint I_;		// number of items

	MatrixKV_<double> beta_;	// word distribution of topic
	MatrixIK_<double> theta_;
	MatrixUK_<double> user_factor_;
	MatrixIK_<double> item_factor_;

	double likelihood_;
	const double conv_epsilon_ = 1e-4;

	// temporary
	VectorK_<double> gamma_;
	MatrixKV_<double> log_beta_;
	MatrixKV_<double> word_ss_;
	MatrixTK_<double> phi_;

private:
	void init();

	void saveTmp() const;
	void save() const;
	void load();

	double docInference(ItemId id, bool update_word_ss);

	void updateU();
	void updateV();
	void updateBeta();

public:
	CTR(uint topic_num, CtrHyperparameter hparam, DocumentSetPtr docs, BooleanMatrix const& ratings) :
		hparam_(hparam), input_data_(docs), tokens_(docs->tokens_), item_token_(docs->getDevidedDocument()),
		user_rating_(ratings.getUsers()), item_rating_(ratings.getItems()), T_(docs->getTokenNum()), K_(topic_num), V_(docs->getWordNum()),
		U_(user_rating_.size()), I_(item_rating_.size()), beta_(K_, V_), theta_(I_, K_), user_factor_(U_, K_), item_factor_(I_, K_), likelihood_(-std::exp(50)),
		gamma_(K_), log_beta_(K_, V_), word_ss_(K_, V_), phi_(T_, K_)
	{
		init();
	}

	void train(uint max_iter, uint min_iter, uint save_lag);
};

}

#endif