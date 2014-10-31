#include "ctr.hpp"
#include "SigUtil/lib/calculation/assign_operation.hpp"

namespace sigtm
{

using namespace boost::numeric;

void saveTmp()
{
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
}

void save()
{
	// save final results
	  sprintf(name, "%s/final-U.dat", directory);
	  FILE * file_U = fopen(name, "w");
	  mtx_fprintf(file_U, user_factor_);
	  fclose(file_U);

	  sprintf(name, "%s/final-V.dat", directory);
	  FILE * file_V = fopen(name, "w");
	  mtx_fprintf(file_V, item_factor_);
	  fclose(file_V);

	  if (hparam_.ctr_run) { 
		sprintf(name, "%s/final-theta.dat", directory);
		FILE * file_theta = fopen(name, "w");
		mtx_fprintf(file_theta, m_theta);
		fclose(file_theta);

		sprintf(name, "%s/final-beta.dat", directory);
		FILE * file_beta = fopen(name, "w");
		mtx_fprintf(file_beta, m_beta);
		fclose(file_beta);
	  }
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

	sig::compound_assign_diagonal(sig::assign_plus(), XX, hparam_.lambda_u_);
	
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
	double delta_ab = hparam_.a_ - hparam_.b_;
	MatrixKK_<double> XX(K_, K_, 0);

	VectorK_<double> gamma;
	MatrixTK_<double> phi;
	MatrixKV_<double> word_ss;
	MatrixKV_<double> log_beta;

	if (hparam_.theta_opt_){
		gamma = VectorK_<double>(K_, 0);
		phi = MatrixTK_<double>(T_, K_, 0);
		word_ss = MatrixKV_<double>(K_, V_, 0);
		log_beta = beta_;
		sig::compound_assign_all([&](double& x, int){ x = x > 0 ? std::log(x) : log_lower_limit; }, log_beta, 0);
	}

	
	for (uint j = 0; j < U_; ++j){
		if (!user_rating_[j].empty()){
			auto const& vec_u = row(user_factor_, j);
			XX += outer_prod(vec_u, vec_u);
		}
	}
	XX *= hparam_.b_;

	for (uint i = 0; i < I_; ++i){
		auto& vec_v = row(item_factor_, i);
		auto const& theta_v = row(theta_, i);
		auto const& user_ids = item_rating_[i];

		if (!user_ids.empty()){
			auto A = XX;
			VectorK_<double> x(K_, 0);

			for (auto user_id : user_ids){
				auto const& vec_u = row(user_factor_, user_id);

				A += delta_ab * outer_prod(vec_u, vec_u);
				x += hparam_.a_ * vec_u;
			}

			x += hparam_.lambda_v_ * theta_v;	// adding the topic vector

			auto B = A;		// save for computing likelihood 

			sig::compound_assign_diagonal(sig::assign_plus(), A, hparam_.lambda_v_);
			vec_v = *sig::matrix_vector_solve(A, std::move(x));	// update vector v

			// update the likelihood for the relevant part
			likelihood_ += -0.5 * item_rating_[i].size() * hparam_.a_;

			for (auto user_id : user_ids){
				auto const& vec_u = row(user_factor_, user_id);
				auto result = inner_prod(vec_u, vec_v);
				likelihood_ += hparam_.a_ * result;
			}
			likelihood_ += -0.5 * mahalanobis_prod(B, &v.vector, &v.vector);

			// likelihood part of theta, even when theta=0, which is a special case
			auto x2 = vec_v;
			x2 -= theta_v;

			auto result = inner_prod(x2, x2);
			likelihood_ += -0.5 * hparam_.lambda_v_ * result;

			if (hparam_.theta_opt_){
				auto const& token_ids = item_token_[i];
				likelihood_ += doc_inference(token_ids, theta_v, log_beta, phi, gamma, word_ss, true);
				optimize_simplex(gamma, vec_v, hparam_.lambda_v_, theta_v);
			}
		}
		else{
			// m=0, this article has never been rated
			if (hparam_.theta_opt_) {
				auto const& token_ids = item_token_[i];
				doc_inference(token_ids, theta_v, log_beta, phi, gamma, word_ss, false);
				vnormalize(gamma);
				theta_v = gamma;
			}
		}
	}
}

void CTR::updateBeta()
{
// update beta if needed
    if (hparam_.ctr_run && hparam_.theta_opt) {
        gsl_matrix_memcpy(m_beta, word_ss);
        for (k = 0; k < m_num_factors; k ++) {
          gsl_vector_view row = gsl_matrix_row(m_beta, k);
          vnormalize(&row.vector);
        }
        gsl_matrix_memcpy(log_beta, m_beta);
        mtx_log(log_beta);
    }
}

void CTR::update(uint max_iter, uint min_iter)
{
	uint iter = 0;
	double likelihood_old_;

	while ((iter < max_iter && converge > 1e-4 ) || iter < min_iter) {
		likelihood_old_ = likelihood_;
		likelihood_ = 0.0;

		updateU();

		if (hparam_.lda_regression_) break; // one iteration is enough for lda-regression

		updateV();
		
		updateBeta();

		time(&current);
		elapsed = (int)difftime(current, start);

		iter++;
		converge = fabs((likelihood-likelihood_old)/likelihood_old);

		if (likelihood < likelihood_old) printf("likelihood is decreasing!\n");

		fprintf(file, "%04d %06d %10.5f %.10f\n", iter, elapsed, likelihood, converge);
		fflush(file);
		printf("iter=%04d, time=%06d, likelihood=%.5f, converge=%.10f\n", iter, elapsed, likelihood, converge);

		// save intermediate results
		if (iter % hparam_.save_lag == 0) {
			saveTmp();
		}
	}
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

  gsl_matrix* phi = NULL;
  gsl_matrix* word_ss = NULL;
  gsl_matrix* log_beta = NULL;
  gsl_vector* gamma = NULL;

  if (hparam_.ctr_run && hparam_.theta_opt) {
    int max_len = c->max_corpus_length();
    phi = gsl_matrix_calloc(max_len, m_num_factors);
    word_ss = gsl_matrix_calloc(m_num_factors, c->m_size_vocab);
    log_beta = gsl_matrix_calloc(m_num_factors, c->m_size_vocab);
    gsl_matrix_memcpy(log_beta, m_beta);
    mtx_log(log_beta);
    gamma = gsl_vector_alloc(m_num_factors);
  }

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