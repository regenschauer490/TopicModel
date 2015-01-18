/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_EIGEN_UTIL_HPP
#define SIGTM_EIGEN_UTIL_HPP

#include "../sigtm.hpp"

#if SIG_USE_EIGEN
#include "SigUtil/lib/helper/container_traits.hpp"
#include "Eigen/Core"

using EigenVector = Eigen::VectorXd;
using EigenMatrix = Eigen::MatrixXd;

namespace sig
{
	namespace impl{
		struct eigen_container_traits
		{
			static const bool exist = true;

			using value_type = double;
		};

		template<>
		struct container_traits<EigenVector> : public eigen_container_traits
		{};
	}
}

#else
#include "SigUtil/lib/calculation/ublas.hpp"
#endif


namespace sigtm
{
#if SIG_USE_EIGEN

template <class M>
auto row_(M&& src, uint i) ->decltype(src.row(i))
{
	return src.row(i);
}
template <class T>
auto row_(std::vector<std::vector<T>>& src, uint i) ->decltype(src[i])
{
	return src[i];
}
template <class T>
auto row_(std::vector<std::vector<T>> const& src, uint i) ->decltype(src[i])
{
	return src[i];
}

template <class M>
auto at_(M&& src, uint row, uint col) ->decltype(src.coeffRef(row, col))
{
	return src.coeffRef(row, col);
}
template <class T>
auto at_(std::vector<std::vector<T>>& src, uint row, uint col) ->decltype(src[row][col])
{
	return src[row][col];
}
template <class T>
auto at_(std::vector<std::vector<T>> const& src, uint row, uint col) ->decltype(src[row][col])
{
	return src[row][col];
}

template <class V>
auto make_zero(uint size)
{
	return V::Zero(size);
}

template <class M>
auto make_zero(uint size_row, uint size_col)
{
	return M::Zero(size_row, size_col);
}

template <class V>
void normalize_dist_v(V&& vec)
{
	double sum = vec.sum();
	vec.array() /= sum;
}

template <class V>
auto sum_v(V const& vec)
{
	return vec.sum();
}

template <class F, class V>
auto map_v(F&& func, V&& vec)
{
	using RT = decltype(sig::impl::eval(std::forward<F>(func), std::forward<V>(vec)(0)));
	EigenVector result(vec.size());

	for (uint i = 0, size = vec.size(); i < size; ++i) {
		result[i] = std::forward<F>(func)(std::forward<V>(vec)(i));
	}
	return result;
}

template <class F, class M>
auto map_m(F&& func, M&& mat)
{
	using RT = decltype(sig::impl::eval(std::forward<F>(func), std::forward<M>(mat)(0, 0)));
	const uint col_size = mat.cols();
	const uint row_size = mat.rows();

	EigenMatrix result(row_size, col_size);

	for (uint i = 0; i < row_size; ++i) {
		for (uint j = 0; j < col_size; ++j) {
			result(i, j) = std::forward<F>(func)(std::forward<M>(mat)(i, j));
		}
	}
	return result;
}

template <class V, class T>
void assign_v(V& vec, T val)
{
	for (uint i = 0, size = vec.size(); i < size; ++i) vec[i] = val;
}

template <class V, class T>
void compound_assign_plus_v(V& vec, T val)
{
	vec.array() += val;
}

template <class V, class T>
void compound_assign_mult_v(V& vec, T val)
{
	vec.array() *= val;
}

template <class V>
auto to_stl_vector(V&& vec)
{
	using T = typename sig::impl::remove_const_reference<decltype(vec[0])>::type;
	const uint size = vec.size();
	std::vector<T> result(size);

	for (uint i = 0; i < size; ++i) {
		result[i] = vec(i);
	}
	return result;
}

template <class M>
auto to_stl_matrix(M&& mat)
{
	using T = typename sig::impl::remove_const_reference<decltype(mat[0])>::type;
	const uint col_size = mat.cols();
	const uint row_size = mat.rows();
	std::vector<std::vector<T>> result(row_size, std::vector<T>(col_size));

	for (uint i = 0; i < row_size; ++i) {
		for (uint j = 0; j < col_size; ++j) {
			result[i][j] = mat(i, j);
		}
	}
	return result;
}

#else
using namespace boost::numeric;

template <class V>
static auto row_(V&& src, uint i) ->decltype(ublas::row(src, i))
{
	return ublas::row(src, i);
}

template <class V>
static auto at_(V&& src, uint row, uint col) ->decltype(src(row, col))
{
	return src(row, col);
}

template <class V>
auto make_zero(uint size)
{
	return V(size, 0);
}

template <class M>
auto make_zero(uint size_row, uint size_col)
{
	return M(size_row, size_col, 0);
}

template <class V>
void normalize_dist_v(V&& vec)
{
	return sig::normalize_dist(vec);
}

template <class V>
auto sum_v(V const& vec)
{
	return sum(vec);
}

template <class F, class V>
auto map_v(F&& func, V&& vec)
{
	return sig::map_v(std::forward<F>(func), std::forward<V>(vec));
}

template <class F, class M>
auto map_m(F&& func, M&& mat)
{
	return sig::map_m(std::forward<F>(func), std::forward<M>(mat));
}

template <class V, class T>
void assign_v(V& vec, T val)
{
	sig::for_each_v([val](double& v) { v = val; }, vec);
}

template <class V, class T>
void compound_assign_plus_v(V& vec, T val)
{
	sig::for_each_v([val](double& v){ v += val; }, vec);
}

template <class V, class T>
void compound_assign_mult_v(V& vec, T val)
{
	sig::for_each_v([val](double& v) { v *= val; }, vec);
}


template <class V>
auto to_stl_vector(V&& vec)
{
	return from_vector_ublas(std::forward<V>(vec));
}

template <class M>
auto to_stl_matrix(M&& mat)
{
	return from_matrix_ublas(std::forward<V>(mat));
}

#endif

}
#endif