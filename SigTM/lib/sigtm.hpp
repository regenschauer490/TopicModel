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

#define USE_SIGNLP 1				// 文字列解析を行うためにSigNLPを使用するか

const bool FixedRandom = true;		// 乱数を固定するか(テスト用)
const std::size_t ThreadNum = 15;	// 並列処理部分で起動するスレッド数

std::wstring const TOKEN_FILENAME = L"token";
std::wstring const VOCAB_FILENAME = L"vocab";
std::wstring const DOC_FILENAME = L"document_names.txt";

const double default_alpha_base = 50;
const double default_beta = 0.1;

using sig::uint;
using sig::StrPtr;
using sig::C_StrPtr;
using sig::WStrPtr;
using sig::C_WStrPtr;
using sig::Maybe;
auto const nothing = boost::none; 
using sig::FilepassString;

using Document = std::vector<std::wstring>;
using Documents = std::vector< std::vector<std::wstring> >;

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
}

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