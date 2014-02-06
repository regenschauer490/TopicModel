#pragma once

#include "helper/SigUtil/lib/sigutil.hpp"
#include "helper/SigUtil/lib/file.hpp"
#include "helper/SigUtil/lib/string.hpp"
#include "helper/SigUtil/lib/eraser.hpp"
#include "helper/SigUtil/lib/tool.hpp"

namespace sigdm{

#define USE_MECAB 0

const bool FIXED_RANDOM = true;		//乱数を固定するか(テスト用)
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

typedef std::vector<std::wstring> Document;
typedef std::vector< std::vector<std::wstring> > Documents;

}	//namespace sigdm