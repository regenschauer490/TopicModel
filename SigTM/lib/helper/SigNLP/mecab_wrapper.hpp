#ifndef SIG_MECAB_WRAPPER_HPP
#define SIG_MECAB_WRAPPER_HPP

#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <memory>
#include <functional>

#include "signlp.hpp"
#include "SigUtil/lib/string/manipulate.hpp"

#if defined(_WIN64)
#include "../../../external/mecab/x64/mecab.h"
#pragma comment(lib, "libmecab.lib")
#elif defined(_WIN32)
#include "../../../external/mecab/x86/mecab.h"
//#pragma comment(lib, "../../SigTM/external/mecab/x86/libmecab.lib")
#else
static_assert(false, "this environment doesn't support.");
#endif


namespace signlp
{
/* MeCab ユーティリティ */
class MecabWrapper
{
	std::shared_ptr<MeCab::Model> model_;

	//ex)もう	副詞,一般,*,*,*,*,もう,モウ,モー,,
	//   眠い	形容詞,自立,*,*,形容詞・アウオ段,基本形,眠い,ネムイ,ネムイ,ねむい/眠い,

	//MeCab::Tagger* _tagger_o;	//ex)もう 眠い
	//_tagger_o(MeCab::createTagger("-Owakati"))

	using TaggerPtr = std::shared_ptr<MeCab::Tagger>;
	using LatticePtr = std::shared_ptr<MeCab::Lattice>;

private:
	MecabWrapper() : model_(MeCab::createModel("")){
			if (!model_) {
				std::cout << "failed to make mecab instance" << std::endl;
				getchar();
			}
		}
	MecabWrapper(MecabWrapper const&) = delete;

	// 並列処理を考慮
	volatile void ParseImpl(std::string const& src, std::string& dest) const;

public:
	static MecabWrapper& getInstance(){
		static MecabWrapper instance;	//thread safe in C++11
		return instance;
	}

	//原文の表現のまま
	auto parseSimple(std::string const& sentence) const->std::vector<std::string>;
	auto parseSimple(std::wstring const& sentence) const->std::vector<std::wstring>;

	//原文の表現のまま + 品詞 <word, word_classs>
	auto parseSimpleWithWC(std::string const& sentence) const->std::vector< std::tuple<std::string, WordClass>>;
	auto parseSimpleWithWC(std::wstring const& sentence) const->std::vector< std::tuple<std::wstring, WordClass>>;

	//原文の表現のまま (pred：品詞選択．trueを返した場合のみコンテナに格納する)
	auto parseSimpleThroughFilter(std::string const& sentence, std::function<bool(WordClass)> const& pred) const->std::vector<std::string>;
	auto parseSimpleThroughFilter(std::wstring const& sentence, std::function<bool(WordClass)> const& pred) const->std::vector<std::wstring>;

	//原形に変換 (skip：原形が存在しないものは無視するか)
	auto parseGenkei(std::string const& sentence, bool skip = true) const->std::vector<std::string>;
	auto parseGenkei(std::wstring const& sentence, bool skip = true) const->std::vector<std::wstring>;

	//原形に変換 + 品詞 <word, word_classs> (skip：原形が存在しないものは無視する)
	auto parseGenkeiWithWC(std::string const& sentence, bool skip = true) const->std::vector< std::tuple<std::string, WordClass>>;
	auto parseGenkeiWithWC(std::wstring const& sentence, bool skip = true) const->std::vector< std::tuple<std::wstring, WordClass>>;
	
	//原形に変形 (pred：品詞選択．trueを返した場合のみコンテナに格納する)
	auto parseGenkeiThroughFilter(std::string const& sentence, std::function<bool(WordClass)> const& pred) const->std::vector<std::string>;
	auto parseGenkeiThroughFilter(std::wstring const& sentence, std::function<bool(WordClass)> const& pred) const->std::vector<std::wstring>;
};


/* implementation */

#define SIG_PARSE_WSTRING_VER(parse_func) \
	std::vector< std::tuple<std::wstring, WordClass> > result; \
	\
	auto tmp = parse_func(sig::wstr_to_str(sentence)); \
	std::transform(tmp.begin(), tmp.end(), std::back_inserter(result), [](std::tuple<std::string, WordClass> const& e){ \
	return std::make_tuple(sig::str_to_wstr(std::get<0>(e)), std::get<1>(e)); \
}); \
	\
	return std::move(result);


inline volatile void MecabWrapper::ParseImpl(std::string const& src, std::string& dest) const
{
	TaggerPtr tagger(model_->createTagger());
	LatticePtr lattice(model_->createLattice());
	/*	if(!tagger || !lattice){
	getchar();
	}*/
	lattice->set_sentence(src.c_str());

	tagger->parse(lattice.get());

	dest.assign(lattice->toString());
}

inline auto MecabWrapper::parseSimple(std::string const& sentence) const ->std::vector<std::string>
{
	std::vector<std::string> result;
	if (enable_warning && sentence.empty()){ std::cout << "sentense is empty" << std::endl; return result; }

	std::string parse;
	ParseImpl(sentence, parse);
	auto wlist = sig::split(parse, "\n");
	for (auto& w : wlist){
		auto tmp = sig::split(w, "\t");
		result.push_back(tmp[0]);
	}

	return std::move(result);
}
inline auto MecabWrapper::parseSimple(std::wstring const& sentence) const->std::vector<std::wstring>
{
	return sig::str_to_wstr(parseSimple(sig::wstr_to_str(sentence)));
}

inline auto MecabWrapper::parseSimpleWithWC(std::string const& sentence) const->std::vector< std::tuple<std::string, WordClass>>
{
	std::vector< std::tuple<std::string, WordClass> > result;
	if (enable_warning && sentence.empty()){ std::cout << "sentense is empty" << std::endl; return result; }

	std::string parse;
	ParseImpl(sentence, parse);
	auto wlist = sig::split(parse, "\n");
	for (auto& w : wlist){
		auto tmp = sig::split(w, "\t");
		if (tmp.size() < 2) continue;
		auto tmp2 = sig::split(tmp[1], ",");
		if (tmp2.size() < 7) continue;
		result.push_back(std::make_tuple(tmp[0], StrToWC(tmp2[0])));
	}

	return std::move(result);
}
inline auto MecabWrapper::parseSimpleWithWC(std::wstring const& sentence) const->std::vector< std::tuple<std::wstring, WordClass>>
{
	SIG_PARSE_WSTRING_VER(parseSimpleWithWC);
}

inline auto MecabWrapper::parseGenkei(std::string const& sentence, bool skip) const->std::vector<std::string>
{
	std::vector<std::string> result;
	if (enable_warning && sentence.empty()){ std::cout << "sentense is empty" << std::endl; return result; }

	std::string parse;
	ParseImpl(sentence, parse);
	auto wlist = sig::split(parse, "\n");

	for (auto& w : wlist){
		auto tmp = sig::split(w, "\t");
		if (tmp.size() < 2) continue;
		auto tmp2 = sig::split(tmp[1], ",");
		if (tmp2.size() < 7) continue;
		if (tmp2[6] == "*"){
			if (!skip) result.push_back(tmp[0]);
		}
		else result.push_back(tmp2[6]);
	}

	return std::move(result);
}
inline auto MecabWrapper::parseGenkei(std::wstring const& sentence, bool skip) const->std::vector<std::wstring>
{
	return sig::str_to_wstr(parseGenkei(sig::wstr_to_str(sentence)));
}

inline auto MecabWrapper::parseGenkeiWithWC(std::string const& sentence, bool skip) const->std::vector< std::tuple<std::string, WordClass>>
{
	std::vector< std::tuple<std::string, WordClass> > result;
	if (enable_warning && sentence.empty()){ std::cout << "sentense is empty" << std::endl; return result; }

	std::string parse;
	ParseImpl(sentence, parse);

	auto wlist = sig::split(parse, "\n");
	for (auto& w : wlist){
		auto tmp = sig::split(w, "\t");
		if (tmp.size() < 2) continue;
		auto tmp2 = sig::split(tmp[1], ",");
		if (tmp2.size() < 7) continue;
		if (tmp2[6] == "*"){
			if (!skip) result.push_back(std::make_tuple(tmp[0], StrToWC(tmp2[0])));
		}
		else result.push_back(std::make_tuple(tmp2[6], StrToWC(tmp2[0])));
	}

	return std::move(result);
}
inline auto MecabWrapper::parseGenkeiWithWC(std::wstring const& sentence, bool skip) const->std::vector< std::tuple<std::wstring, WordClass>>
{
	SIG_PARSE_WSTRING_VER(parseGenkeiWithWC);
}

inline auto MecabWrapper::parseSimpleThroughFilter(std::string const& sentence, std::function<bool(WordClass)> const& pred) const->std::vector<std::string>
{
	auto parse = parseSimpleWithWC(sentence);

	std::vector<std::string> result;
	for (const auto& w : parse){
		if (pred(std::get<1>(w))) result.push_back(std::get<0>(w));
	}

	return std::move(result);
}
inline auto MecabWrapper::parseSimpleThroughFilter(std::wstring const& sentence, std::function<bool(WordClass)> const& pred) const->std::vector<std::wstring>
{
	return sig::str_to_wstr(parseSimpleThroughFilter(sig::wstr_to_str(sentence), pred));
}

inline auto MecabWrapper::parseGenkeiThroughFilter(std::string const& sentence, std::function<bool(WordClass)> const& pred) const->std::vector<std::string>
{
	auto parse = parseGenkeiWithWC(sentence);

	std::vector<std::string> result;
	for (const auto& w : parse){
		if (pred(std::get<1>(w))) result.push_back(std::get<0>(w));
	}

	return std::move(result);
}
inline auto MecabWrapper::parseGenkeiThroughFilter(std::wstring const& sentence, std::function<bool(WordClass)> const& pred) const->std::vector<std::wstring>
{
	return sig::str_to_wstr(parseGenkeiThroughFilter(sig::wstr_to_str(sentence), pred));
}


}
#endif

