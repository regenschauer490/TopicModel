/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#include "ctr.h"
#include "SigUtil/lib/calculation/binary_operation.hpp"
#include "SigUtil/lib/tools/convergence.hpp"
#include "SigUtil/lib/functional/filter.hpp"

namespace sigtm
{
using namespace boost::numeric;

const double projection_z = 1.0;

static double safe_log(double x)
{
	return x > 0 ? std::log(x) : log_lower_limit;
};

template <class C>
static auto row_(C&& src, uint i) ->decltype(src[i])
{
	return src[i];
}
template <class C, class D = void>
static auto row_(C&& src, uint i) ->decltype(ublas::row(src, i))
{
	return ublas::row(src, i);
}

template <class C>
static auto at_(C&& src, uint row, uint col) ->decltype(src[row][col])
{
	return src[row][col];
}
template <class C, class D = void>
static auto at_(C&& src, uint row, uint col) ->decltype(src(row, col))
{
	return src(row, col);
}

template <class V>
bool is_feasible(V const& x)
{
	double val;
	double sum = 0;
	for (uint i = 0, size = x.size()-1; i < size; ++i) {
		val = x[i];
		if (val < 0 || val >1) return false;
		sum += val;
		if (sum > 1) return false;
	}
	return true;
}

// project x on to simplex (using // http://www.cs.berkeley.edu/~jduchi/projects/DuchiShSiCh08.pdf)
template <class V1, class V2>
void simplex_projection(
	V1 const& x,
	V2& x_proj,
	double z)
{
	x_proj = x;
	std::sort(x_proj.begin(), x_proj.end());
	double cumsum = -z, u;
	int j = 0;
	
	for (int i = x.size() - 1; i >= 0; --i) {
		u = x_proj[i];
		cumsum += u;
		if (u > cumsum / (j + 1)) j++;
		else break;
	}
	double theta = cumsum / j;
	for (int i = 0, size = x.size(); i < size; ++i) {
		u = x[i] - theta;
		if (u <= 0) u = 0.0;
		x_proj[i] = u;
	}
	sig::normalize_dist_v(x_proj); // fix the normaliztion issue due to numerical errors
}

template <class V1, class V2, class V3>
auto df_simplex(
	V1 const&gamma,
	V2 const& v,
	double lambda,
	V3 const& opt_x)
{
	sig::vector_u<double> g = -lambda * (opt_x - v);
	sig::vector_u<double> y = gamma;

	sig::for_each_v([](double& v1, double v2){ v1 /= v2; }, y, opt_x);
	g += y;
	sig::for_each_v([](double& v1){ v1 *= -1; }, g);

	return g;
}

template <class V1, class V2, class V3>
double f_simplex(
	V1 const& gamma,
	V2 const& v,
	double lambda,
	V3 const& opt_x)
{
	auto y = sig::map_v([&](double x){ return safe_log(x); }, opt_x);
	auto z = v - opt_x;
	
	double f = ublas::inner_prod(y, gamma);
	double val = ublas::inner_prod(z, z);
	f -= 0.5 * lambda * val;

	return -f;
}

// projection gradient algorithm
template <class V1, class V2, class V3>
void optimize_simplex(
	V1 const& gamma, 
	V2 const& v, 
	double lambda,
	V3& opt_x)
{
	size_t size = sig::min(gamma.size(), v.size());
	sig::vector_u<double> x_bar(size);
	sig::vector_u<double> opt_x_old = opt_x;

	double f_old = f_simplex(gamma, v, lambda, opt_x);

	auto g = df_simplex(gamma, v, lambda, opt_x);

	sig::normalize_dist_v(g);
	//double ab_sum = sig::sum(g);
	//if (ab_sum > 1.0) g *= (1.0 / ab_sum); // rescale the gradient

	opt_x -= g;

	simplex_projection(opt_x, x_bar, projection_z);

	x_bar -= opt_x_old;
	
	double r = 0.5 * ublas::inner_prod(g, x_bar);

	const double beta = 0.5;
	double t = beta;
	for (uint iter = 0; iter < 100; ++iter) {
		opt_x = opt_x_old;
		opt_x += t * x_bar;

		double f_new = f_simplex(gamma, v, lambda, opt_x);

		if (f_new > f_old + r * t) t = t * beta;
		else break;
	}

	if (!is_feasible(opt_x))  printf("sth is wrong, not feasible. you've got to check it ...\n");
}


void CTR::init()
{
	sig::SimpleRandom<double> randf(0, 1, FixedRandom);

	beta_ = MatrixKV_<double>(K_, V_); //SIG_INIT_MATRIX(double, K, V, 0);

	for(TopicId k = 0; k < K_; ++k){
		auto& beta_v = row_(beta_, k);
		for (auto& e : beta_v) e = randf() + hparam_.beta_smooth_;
		sig::normalize_dist_v(beta_v);
	}


	theta_ = MatrixIK_<double>(I_, K_, 0); // SIG_INIT_MATRIX(double, I, K, 0);

	if (hparam_.lda_regression_){
		for (ItemId i = 0; i < I_; ++i){
			auto& theta_v = row_(theta_, i);
			for (auto& e :theta_v) e = randf() + hparam_.alpha_smooth_;
			sig::normalize_dist_v(theta_v);
		}
	}

	user_factor_ = MatrixUK_<double>(U_, K_, 0);
	item_factor_ = MatrixIK_<double>(I_, K_, 0);

	for (ItemId i = 0; i < I_; ++i){
		for (auto& e : row_(item_factor_, i)) e = randf();
	}
}


void CTR::printUFactor() const
{
	std::cout << "user_factor" << std::endl;
	for (uint u = 0; u<U_; ++u){
		for (uint k = 0; k<K_; ++k) std::cout << user_factor_(u, k) << ", ";
		std::cout << std::endl;
	}
}
void CTR::printIFactor() const
{
	std::cout << "item_factor" << std::endl;
	for (uint i = 0; i<I_; ++i){
		for (uint k = 0; k<K_; ++k) std::cout << item_factor_(i, k) << ", ";
		std::cout << std::endl;
	}
}

/*
	std::cout << "estimate rating" << std::endl;
	for (uint u = 0; u<ratings_->userSize(); ++u){
		for (uint i = 0; i<ratings_->itemSize(); ++i) std::cout << estimate(u, i) << ", ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
*/

void CTR::saveTmp() const
{
	/*
	sprintf(name, "%s/%04d-U.dat", directory, iter);
      FILE * file_U = fopen(name, "w");
      mtx_fprintf(file_U, user_factor_);
      fclose(file_U);

      sprintf(name, "%s/%04d-V.dat", directory, iter);
      FILE * file_V = fopen(name, "w");
      mtx_fprintf(file_V, item_factor_);
      fclose(file_V);

      if (hparam_.ctr_run) { 
        sprintf(name, "%s/%04d-theta.dat", directory, iter);
        FILE * file_theta = fopen(name, "w");
        mtx_fprintf(file_theta, m_theta);
        fclose(file_theta);

        sprintf(name, "%s/%04d-beta.dat", directory, iter);
        FILE * file_beta = fopen(name, "w");
        mtx_fprintf(file_beta, m_beta);
        fclose(file_beta);
	}
	*/
}

void CTR::save() const
{
}

void CTR::load()
{

}

double CTR::docInference(ItemId id,	bool update_word_ss)
{
	double pseudo_count = 1.0;
	double likelihood = 0;
	auto const& theta_v = row_(theta_, id);
	auto log_theta_v = sig::map_v([&](double x){ return safe_log(x); }, theta_v);
	
	for (auto tid : item_tokens_[id]){
		WordId w = tokens_[tid].word_id;
		auto& phi_v = row_(phi_, tid);

		for (TopicId k = 0; k < K_; ++k){
			phi_v[k] = theta_v[k] * at_(beta_, k, w);
		}
		sig::normalize_dist_v(phi_v);

		for (TopicId k = 0; k < K_; ++k){
			double const& p = phi_v[k];
			if (p > 0){
				likelihood += p * (log_theta_v[k] + log_beta_(k, w) - std::log(p));
			}
		}
	}

	if (pseudo_count > 0) {
		likelihood += pseudo_count * std::accumulate(std::begin(log_theta_v), std::end(log_theta_v), 0.0);
	}

	// smoothing with small pseudo counts
	sig::for_each_v([&](double& v){ v = pseudo_count; }, gamma_);
	
	for (auto tid : item_tokens_[id]){
		for (TopicId k = 0; k < K_; ++k) {
			//double x = doc->m_counts[tid] * phi_(tid, k);	// doc_word_ct only
			double const& x = at_(phi_, tid, k);
			gamma_[k] += x;
			
			if (update_word_ss){
				at_(word_ss_, k, tokens_[tid].word_id) -= x;
			}
		}
	}

	return likelihood;
}

void CTR::updateU()
{ 
	double delta_ab = hparam_.a_ - hparam_.b_;
	MatrixKK_<double> XX(K_, K_, 0);

	// calculate VCV^T in equation(8)
	for (uint i = 0; i < I_; i ++){
		if (std::begin(item_ratings_[i]) != std::end(item_ratings_[i])){
			auto const& vec_v = row_(item_factor_, i);

			XX += outer_prod(vec_v, vec_v);
		}
    }
	
	// negative item weight
	XX *= hparam_.b_;

	sig::for_diagonal([&](double& v){ v += hparam_.lambda_u_; }, XX);
		
	for (uint j = 0; j < U_; ++j){
		auto const& ratings = user_ratings_[j];

		if (std::begin(ratings) != std::end(ratings)){
			auto A = XX;
			VectorK_<double> x(K_, 0);

			for (auto rating : ratings){
				auto const& vec_v = row_(item_factor_, rating->item_id_);

				A += delta_ab * outer_prod(vec_v, vec_v);
				x += hparam_.a_ * vec_v;
			}

			auto& vec_u = row(user_factor_, j);
			vec_u = *sig::matrix_vector_solve(std::move(A), std::move(x));	// update vector u

			// update the likelihood
			auto result = inner_prod(vec_u, vec_u);
			likelihood_ += -0.5 * hparam_.lambda_u_ * result;
		}
	}
}

void CTR::updateV()
{
	auto mahalanobis_prod = [&](sig::matrix_u<double> const& m, sig::vector_u<double> const& v)
	{
		return inner_prod(v, prod(m, v));
	};

	double delta_ab = hparam_.a_ - hparam_.b_;
	MatrixKK_<double> XX(K_, K_, 0);
	
	for (uint j = 0; j < U_; ++j){
		if (std::begin(user_ratings_[j]) != std::end(user_ratings_[j])){
			auto const& vec_u = row(user_factor_, j);
			XX += outer_prod(vec_u, vec_u);
		}
	}
	XX *= hparam_.b_;
	
	for (uint i = 0; i < I_; ++i){
		auto& vec_v = row_(item_factor_, i);
		auto const& theta_v = row_(theta_, i);
		auto const& ratings = item_ratings_[i];

		if (std::begin(ratings) != std::end(ratings)){
			auto A = XX;
			VectorK_<double> xx(K_, 0);

			for (auto rating : ratings){
				auto const& vec_u = row_(user_factor_, rating->user_id_);

				A += delta_ab * outer_prod(vec_u, vec_u);
				xx += hparam_.a_ * vec_u;
			}

			// xx += hparam_.lambda_v_ * theta_v;	// adding the topic vector
			sig::for_each_v([&](double& x, double t){ x += hparam_.lambda_v_ * t; }, xx, theta_v);

		
			auto B = A;		// save for computing likelihood 

			sig::for_diagonal([&](double& v){ v += hparam_.lambda_v_; }, A);
			vec_v = *sig::matrix_vector_solve(A, std::move(xx));	// update vector v

			// update the likelihood for the relevant part
			likelihood_ += -0.5 * item_ratings_[i].size() * hparam_.a_;


			for (auto rating : ratings){
				auto const& vec_u = row_(user_factor_, rating->user_id_);
				auto result = inner_prod(vec_u, vec_v);
				likelihood_ += hparam_.a_ * result;
			}
			likelihood_ += -0.5 * mahalanobis_prod(B, vec_v);

			// likelihood part of theta, even when theta=0, which is a special case
			sig::vector_u<double> x2 = vec_v;
			
			//x2 -= theta_v;
			sig::for_each_v([](double& v1, double v2){ v1 -= v2; }, x2, theta_v);

			auto result = inner_prod(x2, x2);
			likelihood_ += -0.5 * hparam_.lambda_v_ * result;

			if (hparam_.theta_opt_){
				likelihood_ += docInference(i, true);
				optimize_simplex(gamma_, vec_v, hparam_.lambda_v_, row_(theta_, i));
			}
		}
		else{
			// m=0, this article has never been rated
			if (hparam_.theta_opt_) {
				docInference(i, false);
				sig::normalize_dist_v(gamma_);
				row(theta_, i) = gamma_;
			}
		}
	}
}

void CTR::updateBeta()
{
	beta_ = word_ss_;

	for (TopicId k = 0; k < K_; ++k){
		auto& beta_v = row_(beta_, k);

		sig::normalize_dist_v(beta_v);
		row_(log_beta_, k) = sig::map_v([&](double x){ return safe_log(x); }, beta_v);
	}
}


void CTR::train(uint max_iter, uint min_iter, uint save_lag)
{
	uint iter = 0;
	double likelihood_old;
	sig::ManageConvergenceSimple conv(conv_epsilon_);
	
	auto info_print = [](uint iter, double likelihood, double converge){
		std::cout << "iter=" << iter << ", likelihood=" << likelihood << ", converge=" << converge << std::endl;
		return true;
	};

	if (max_iter < min_iter) std::swap(max_iter, min_iter);

	if (hparam_.theta_opt_){
		gamma_ = VectorK_<double>(K_, 0);
		log_beta_ = sig::map_m([&](double x){ return safe_log(x); }, beta_);
		word_ss_ = MatrixKV_<double>(K_, V_); // SIG_INIT_MATRIX(double, K, V, 0);
		phi_ = MatrixKV_<double>(T_, K_);  //SIG_INIT_MATRIX(double, T, K, 0);
	}

	 while ((!conv.is_convergence() && iter < max_iter) || iter < min_iter)
	 {
		likelihood_old = likelihood_;
		likelihood_ = 0.0;

		//printUFactor();
		//printIFactor();

		updateU();
		
		//if (hparam_.lda_regression_) break; // one iteration is enough for lda-regression

		updateV();
		
		// update beta if needed
		if (hparam_.theta_opt_) updateBeta();

		if(likelihood_ < likelihood_old) std::cout << "likelihood is decreasing!" << std::endl;
		
		// save intermediate results
		if (iter % save_lag == 0) {
			saveTmp();
		}

		++iter;
		conv.update( sig::abs_delta(likelihood_, likelihood_old) / likelihood_old);

		info_print(iter, likelihood_, conv.get_value());
	 }

	 std::cout << "train finished" << std::endl;
}

auto CTR::recommend(Id id, bool for_user, double threshold) const->std::vector<Id>
{
	std::vector<Id> result;
	
	if (for_user){
		for (uint i = 0; i < I_; ++i){
			if(estimate(id, i) > threshold) result.push_back(i);
		}
	}
	else{
		for (uint u = 0; u < U_; ++u){
			if (estimate(u, id) > threshold) result.push_back(u);
		}
	}

	sig::sort(result, std::less<double>());

	return result;
}

auto CTR::recommend_detail(Id id, bool for_user, double threshold) const->std::vector<EstRatingPtr_>
{
	auto tmp = for_user
		? sig::map([&](uint i){ return std::make_shared<Rating_<EstValueType>>(id, i, estimate(id, i)); }, sig::seqn(0, 1, I_))
		: sig::map([&](uint u){ return std::make_shared<Rating_<EstValueType>>(u, id, estimate(u, id)); }, sig::seqn(0, 1, U_));

	auto result = sig::filter([&](EstRatingPtr_ const& e){ return e->value_ > threshold; }, tmp);
	sig::sort(result, [](EstRatingPtr_ const& v1, EstRatingPtr_ const& v2){ return v1->value_ > v2->value_; });

	return result;
}

inline double CTR::estimate(UserId u_id, ItemId i_id) const
{
	return inner_prod(row_(user_factor_, u_id), row_(item_factor_, i_id));
}

/*
void c_ctr::learn_map_estimate(
	const c_data* users,
	const c_data* items,
	const c_corpus* c,
	const ctr_hyperparameter* param,
	const char* directory)
{
  // init model parameters
  printf("\ninitializing the model ...\n");
  init_model(hparam_.ctr_run);

  // filename
  char name[500];

  // start time
  time_t start, current;
  time(&start);
  int elapsed = 0;

  int iter = 0;
  double likelihood = -exp(50), likelihood_old;
  double converge = 1.0;

  /// create the state log file 
  sprintf(name, "%s/state.log", directory);
  FILE* file = fopen(name, "w");
  fprintf(file, "iter time likelihood converge\n");

  int i, j, m, n, l, k;
  int* item_ids; 
  int* user_ids;

  double result;

  /// confidence parameters
  double a_minus_b = hparam_.a - hparam_.b;

  
  update();  
 
  save();

  // free memory
  gsl_matrix_free(XX);
  gsl_matrix_free(A);
  gsl_matrix_free(B);
  gsl_vector_free(x);

  if (hparam_.ctr_run && hparam_.theta_opt) {
    gsl_matrix_free(phi);
    gsl_matrix_free(log_beta);
    gsl_matrix_free(word_ss);
    gsl_vector_free(gamma);
  }
}
*/

}	// sigtm