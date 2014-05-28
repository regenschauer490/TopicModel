/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_TM_HPP
#define SIG_TM_HPP

#include "SigUtil/lib/sigutil.hpp"

namespace sigtm
{
#define USE_SIGNLP 0				// �������͂��s�����߂�SigNLP���g�p���邩

const bool FIXED_RANDOM = true;		// �������Œ肷�邩(�e�X�g�p)
const std::size_t THREAD_NUM = 15;

std::wstring const TOKEN_FILENAME = L"token";
std::wstring const VOCAB_FILENAME = L"vocab";


using sig::uint;
using sig::StrPtr;
using sig::C_StrPtr;
using sig::WStrPtr;
using sig::C_WStrPtr;
using sig::maybe;
using sig::nothing;
using sig::FilepassString;

using Document = std::vector<std::wstring>;
using Documents = std::vector< std::vector<std::wstring> >;

}
#endif