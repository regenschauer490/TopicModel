/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_HPP
#define SIGTM_HPP

#include "SigUtil/lib/sigutil.hpp"
#include "helper/SigNLP/signlp.hpp"
#include "SigUtil/lib/helper/container_traits.hpp"

namespace sigtm
{

#define SIG_USE_EIGEN 1			// 行列演算にライブラリのEigenを使用するか（処理速度向上）
#define SIG_USE_SIGNLP 1		// 文字列解析を行うためにSigNLPを使用するか

const bool FixedRandom = true;		// 乱数を固定するか(テスト用)
//const std::size_t ThreadNum = 15;	// 並列処理部分で起動するスレッド数

std::wstring const TOKEN_FILENAME = L"token";
std::wstring const VOCAB_FILENAME = L"vocab";
std::wstring const DOC_FILENAME = L"document_names.txt";

const double default_alpha_base = 50;
const double default_beta = 0.01;

using sig::uint;
using sig::StrPtr;
using sig::C_StrPtr;
using sig::WStrPtr;
using sig::C_WStrPtr;
using sig::Maybe;
auto const nothing = boost::none; 
using sig::FilepassString;

using Text = std::wstring;
using Document = std::vector<Text>;
using Documents = std::vector<Document>;

using DocumentId = uint;
using TopicId = uint;
using WordId = uint;
using UserId = uint;
using TokenId = uint;
using Id = uint;

template<class T> using VectorT = std::vector<T>;	// token
template<class T> using VectorD = std::vector<T>;	// document
template<class T> using VectorK = std::vector<T>;	// topic
template<class T> using VectorV = std::vector<T>;	// word
template<class T> using VectorU = std::vector<T>;	// user

const uint zero = 0;
const double log_lower_limit = -100000;


#if SIG_USE_EIGEN

template <class M>
auto row_(M&& src, uint i) ->decltype(src.row(i))
{
	return src.row(i);
}
template <class T, class D = void>
auto row_(std::vector<std::vector<T>>& src, uint i) ->decltype(src[i])
{
	return src[i];
}
template <class T, class D = void>
auto row_(std::vector<std::vector<T>> const& src, uint i) ->decltype(src[i])
{
	return src[i];
}

template <class M>
auto at_(M&& src, uint row, uint col) ->decltype(src.coeffRef(row, col))
{
	return src.coeffRef(row, col);
}
template <class M, class D = void>
auto at_(M&& src, uint row, uint col) ->decltype(src[row][col])
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
#endif
}


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
#endif

namespace std
{
template <> struct hash<sig::C_WStrPtr>
{
	size_t operator()(sig::C_WStrPtr const& x) const
	{
		return hash<std::wstring>()(*x);
	}
};
}	//std

#endif