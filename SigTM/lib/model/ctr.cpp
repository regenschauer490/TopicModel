#include "ctr.h"
#include "SigUtil/lib/calculation/assign_operation.hpp"
#include "SigUtil/lib/calculation/binary_operation.hpp"
#include "SigUtil/lib/tools/convergence.hpp"

namespace sigtm
{
using namespace boost::numeric;

auto safe_log = [&](double x)
{
	return x > 0 ? std::log(x) : log_lower_limit;
};


bool is_feasible(sig::vector_u<double> const& x)
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
void simplex_projection(
	sig::vector_u<double> const& x,
	sig::vector_u<double>& x_proj,
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
	sig::normalize_dist(x_proj); // fix the normaliztion issue due to numerical errors
}

auto df_simplex(
	sig::vector_u<double> const&gamma,
	sig::vector_u<double> const& v,
	double lambda,
	sig::vector_u<double> const& opt_x)
->sig::vector_u<double>
{
	sig::vector_u<double> g = -lambda * (opt_x - v);
	sig::vector_u<double> y = gamma;

	sig::compound_assign_all(sig::assign_div_t(), y, opt_x);
	g += y;

	return sig::minus(g, -1);
}

double f_simplex(
	sig::vector_u<double> const& gamma,
	sig::vector_u<double> const& v,
	double lambda,
	sig::vector_u<double> const& opt_x)
{
	auto y = sig::map([&](double x){ return safe_log(x); }, opt_x);
	auto z = v - opt_x;
	
	double f = ublas::inner_prod(y, gamma);
	double val = ublas::inner_prod(z, z);
	f -= 0.5 * lambda * val;

	return -f;
}

// projection gradient algorithm
auto optimize_simplex(
	sig::vector_u<double> const& gamma, 
	sig::vector_u<double> const& v, 
	double lambda,
	VectorK<double>& opt_x)
->sig::vector_u<double>
{
	size_t size = sig::min(gamma.size(), v.size());
	sig::vector_u<double> x_bar(size);
	sig::vector_u<double> opt_x_old = opt_x;

	double f_old = f_simplex(gamma, v, lambda, opt_x);

	auto g = df_simplex(gamma, v, lambda, opt_x);

	double ab_sum = sig::sum(g);
	if (ab_sum > 1.0) g *= (1.0 / ab_sum); // rescale the gradient

	opt_x -= g;

	simplex_projection(opt_x, x_bar);

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

	beta_ = SIG_INIT_MATRIX(double, K, V, 0);

	for(TopicId k = 0; k < K_; ++k){
		sig::compound_assignment([&](double& v, int){ v = randf() + hparam_.beta_smooth_; }, beta_[k], 0);
		sig::normalize_dist(beta_[k]);
	}


	theta_ = SIG_INIT_MATRIX(double, I, K, 0);

	for (ItemId i = 0; i < I_; ++i){
		sig::compound_assignment([&](double& v, int){ v = randf() + hparam_.alpha_smooth_; }, theta_[i], 0);
		sig::normalize_dist(theta_[i]);
	}

	item_factor_ = sig::to_matrix_ublas(theta_);
	user_factor_ = MatrixUK_<double>(U_, K_);
}

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
	auto const& theta_v = theta_[id];
	VectorK<double> log_theta_v = sig::map([&](double x){ return safe_log(x); }, theta_v);
	
	for (auto tid : item_token_[id]){
		WordId w = tokens_[tid].word_id;

		for (TopicId k = 0; k < K_; ++k){
			phi_[tid][k] = theta_v[k] * beta_[k][w];
		}
		sig::normalize_dist(phi_[tid]);

		for (TopicId k = 0; k < K_; ++k){
			double const& p = phi_[tid][k];
			if (p > 0){
				likelihood += p * (log_theta_v[k] + log_beta_(k, w) - std::log(p));
			}
		}
	}

	if (pseudo_count > 0) {
		likelihood += pseudo_count * sig::sum(log_theta_v);
	}

	// smoothing with small pseudo counts
	sig::compound_assign_all([](double& v, double x){ v = x; }, gamma_, pseudo_count);
	
	for (auto tid : item_token_[id]){
		for (TopicId k = 0; k < K_; ++k) {
			//double x = doc->m_counts[tid] * phi_(tid, k);	// doc_word_ct only
			double const& x = phi_[tid][k];
			gamma_[k] += x;
			
			if (update_word_ss){
				word_ss_[k][tokens_[tid].word_id] -= x;
			}
		}
	}

	return likelihood;
}

void CTR::updateU()
{ 
	double delta_ab = hparam_.a_ - hparam_.b_;
	MatrixKK_<double> XX(K_, K_, 0);

	for (uint i = 0; i < I_; i ++){
		if (!item_rating_[i].empty()){
			auto const& vec_v = row(item_factor_, i);

			XX += outer_prod(vec_v, vec_v);
		}
    }

	XX *= hparam_.b_;

	sig::compound_assign_diagonal(sig::assign_plus_t(), XX, hparam_.lambda_u_);
	
	for (uint j = 0; j < U_; ++j){
		auto const& item_ids = user_rating_[j];

		if (!user_rating_[j].empty()){
			auto A = XX;
			VectorK_<double> x(K_, 0);

			for (auto item_id : item_ids){
				auto const& vec_v = row(item_factor_, item_id);

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
		if (!user_rating_[j].empty()){
			auto const& vec_u = row(user_factor_, j);
			XX += outer_prod(vec_u, vec_u);
		}
	}
	XX *= hparam_.b_;

	for (uint i = 0; i < I_; ++i){
		auto& vec_v = row(item_factor_, i);
		auto const& theta_v = theta_[i];
		auto const& user_ids = item_rating_[i];

		if (!user_ids.empty()){
			auto A = XX;
			VectorK_<double> xx(K_, 0);

			for (auto user_id : user_ids){
				auto const& vec_u = row(user_factor_, user_id);

				A += delta_ab * outer_prod(vec_u, vec_u);
				xx += hparam_.a_ * vec_u;
			}

			//xx += hparam_.lambda_v_ * theta_v;	// adding the topic vector
			sig::compound_assign_all([&](double& x, double t){ x += hparam_.lambda_v_ * t; }, xx, theta_v);

			auto B = A;		// save for computing likelihood 

			sig::compound_assign_diagonal(sig::assign_plus_t(), A, hparam_.lambda_v_);
			vec_v = *sig::matrix_vector_solve(A, std::move(xx));	// update vector v

			// update the likelihood for the relevant part
			likelihood_ += -0.5 * item_rating_[i].size() * hparam_.a_;

			for (auto user_id : user_ids){
				auto const& vec_u = row(user_factor_, user_id);
				auto result = inner_prod(vec_u, vec_v);
				likelihood_ += hparam_.a_ * result;
			}
			likelihood_ += -0.5 * mahalanobis_prod(B, vec_v);

			// likelihood part of theta, even when theta=0, which is a special case
			sig::vector_u<double> x2 = vec_v;
			//x2 -= theta_v;
			sig::compound_assign_all(sig::assign_minus_t(), x2, theta_v);

			auto result = inner_prod(x2, x2);
			likelihood_ += -0.5 * hparam_.lambda_v_ * result;

			if (hparam_.theta_opt_){
				likelihood_ += docInference(i, true);
				row(item_factor_, i) = optimize_simplex(gamma_, vec_v, hparam_.lambda_v_, theta_v);
			}
		}
		else{
			// m=0, this article has never been rated
			if (hparam_.theta_opt_) {
				docInference(i, false);
				sig::normalize_dist(gamma_);
				row(item_factor_, i) = gamma_;
			}
		}
	}
}

void CTR::updateBeta()
{
	sig::compound_assignment([](double& b, double ws){ b = ws; }, beta_, word_ss_);

	for (auto const& dist : beta_){
		sig::normalize_dist(dist);
	}
	log_beta_ = sig::map([&](double x){ return safe_log(x); }, beta_);
}

void CTR::update(uint max_iter, uint min_iter, uint save_lag)
{
	uint iter = 0;
	double likelihood_old_;
	sig::ManageConvergenceSimple conv(conv_epsilon_);
	
	if (hparam_.theta_opt_){
		gamma_ = VectorK_<double>(K_, 0);
		log_beta_ = sig::map([&](double x){ return safe_log(x); }, beta_);
		word_ss_ = SIG_INIT_MATRIX(double, K, V, 0);
		phi_ = SIG_INIT_MATRIX(double, T, K, 0);
	}

	 do{
		likelihood_old_ = likelihood_;
		likelihood_ = 0.0;

		updateU();

		if (hparam_.lda_regression_) break; // one iteration is enough for lda-regression

		updateV();
		
		// update beta if needed
		if (hparam_.theta_opt_) updateBeta();
		
		if (likelihood < likelihood_old) printf("likelihood is decreasing!\n");

		fprintf(file, "%04d %06d %10.5f %.10f\n", iter, elapsed, likelihood, converge);
		fflush(file);
		printf("iter=%04d, time=%06d, likelihood=%.5f, converge=%.10f\n", iter, elapsed, likelihood, converge);

		// save intermediate results
		if (iter % save_lag == 0) {
			saveTmp();
		}	
	 }
	 while (++iter, (!conv.update(sig::abs_delta(likelihood_, likelihood_old_) / likelihood_old_) && iter < max_iter) || iter < min_iter)
}

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


  /* alloc auxiliary variables */
  gsl_matrix* XX = gsl_matrix_alloc(m_num_factors, m_num_factors);
  gsl_matrix* A  = gsl_matrix_alloc(m_num_factors, m_num_factors);
  gsl_matrix* B  = gsl_matrix_alloc(m_num_factors, m_num_factors);
  gsl_vector* x  = gsl_vector_alloc(m_num_factors);


  /* tmp variables for indexes */
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

}	// sigtm