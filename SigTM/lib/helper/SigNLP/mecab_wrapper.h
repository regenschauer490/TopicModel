#ifndef SIG_MECAB_WRAPPER_H
#define SIG_MECAB_WRAPPER_H

#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <memory>
#include <functional>

#include "signlp.hpp"

#if defined(_WIN64)
#include "../external/mecab/x64/mecab.h"
#pragma comment(lib, "../external/mecab/x64/libmecab.lib")
#elif defined(_WIN32)
#include "../external/mecab/x86/mecab.h"
#pragma comment(lib, "external/mecab/x86/libmecab.lib")
#else
static_assert(false, "this environment doesn't support.");
#endif


namespace signlp{

/* MeCab ラッパ */
class MecabWrapper
{
	MeCab::Model* model_;

	//ex)もう	副詞,一般,*,*,*,*,もう,モウ,モー,,
	//   眠い	形容詞,自立,*,*,形容詞・アウオ段,基本形,眠い,ネムイ,ネムイ,ねむい/眠い,

	//MeCab::Tagger* _tagger_o;	//ex)もう 眠い
	//_tagger_o(MeCab::createTagger("-Owakati"))

	using TaggerPtr = std::shared_ptr<MeCab::Tagger>;
	using LatticePtr = std::shared_ptr<MeCab::Lattice>;

private:
	MecabWrapper() : model_(MeCab::createModel("")){}
	MecabWrapper(MecabWrapper const&) = delete;

	// 並列処理を考慮
	volatile void ParseImpl(std::string const& src, std::string& dest) const;

public:
	static MecabWrapper& getInstance(){
		static MecabWrapper instance;	//thread safe in C++11
		return instance;
	}

	//原文の表現のまま
	std::vector<std::string> parseSimple(std::string const& sentence) const;
	std::vector<std::wstring> parseSimple(std::wstring const& sentence) const;

	//原文の表現のまま + 品詞 <word, word_classs>
	std::vector< std::tuple<std::string, WordClass> > parseSimpleWithWC(std::string const& sentence) const;
	std::vector< std::tuple<std::wstring, WordClass> > parseSimpleWithWC(std::wstring const& sentence) const;

	//原形に変換 (skip：原形が存在しないものは無視するか)
	std::vector<std::string> parseGenkei(std::string const& sentence, bool skip = true) const;
	std::vector<std::wstring> parseGenkei(std::wstring const& sentence, bool skip = true) const;

	//原形に変換 + 品詞 <word, word_classs> (skip：原形が存在しないものは無視する)
	std::vector< std::tuple<std::string, WordClass> > parseGenkeiWithWC(std::string const& sentence, bool skip = true) const;
	std::vector< std::tuple<std::wstring, WordClass> > parseGenkeiWithWC(std::wstring const& sentence, bool skip = true) const;

	//原文の表現のまま (pred：品詞選択．trueを返した場合のみコンテナに格納する)
	std::vector<std::string> parseSimpleThroughFilter(std::string const& sentence, std::function<bool(WordClass)> const& pred) const;
	std::vector<std::wstring> parseSimpleThroughFilter(std::wstring const& sentence, std::function<bool(WordClass)> const& pred) const;

	//原形に変形 (pred：品詞選択．trueを返した場合のみコンテナに格納する)
	std::vector<std::string> parseGenkeiThroughFilter(std::string const& sentence, std::function<bool(WordClass)> const& pred) const;
	std::vector<std::wstring> parseGenkeiThroughFilter(std::wstring const& sentence, std::function<bool(WordClass)> const& pred) const;
};

}
#endif

