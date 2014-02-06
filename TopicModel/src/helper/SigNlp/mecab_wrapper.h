#ifndef __MECAB_WRAPPER_H__
#define __MECAB_WRAPPER_H__

#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <memory>
#include <functional>

#include "signlp.hpp"

#ifdef _WIN64
#include "mecab/x64/mecab.h"
#pragma comment(lib, "mecab/x64/libmecab.lib")
#else
#include "mecab/x86/mecab.h"
#pragma comment(lib, "mecab/x86/libmecab.lib")
#endif


namespace signlp{

/* MeCab ラッパ */
class MecabWrapper
{
	MeCab::Model* _model;

	//ex)もう	副詞,一般,*,*,*,*,もう,モウ,モー,,
	//   眠い	形容詞,自立,*,*,形容詞・アウオ段,基本形,眠い,ネムイ,ネムイ,ねむい/眠い,

	//MeCab::Tagger* _tagger_o;	//ex)もう 眠い
	//_tagger_o(MeCab::createTagger("-Owakati"))

	typedef std::shared_ptr<MeCab::Tagger> TaggerPtr;
	typedef std::shared_ptr<MeCab::Lattice> LatticePtr;

private:
	MecabWrapper() : _model(MeCab::createModel("")){}
	MecabWrapper(MecabWrapper const&);

	volatile void ParseImpl(std::string const& src, std::string& dest) const;

public:
	static MecabWrapper& GetInstance(){
		static MecabWrapper instance;	//thread safe in C++11
		return instance;
	}

	//原文の表現のまま
	std::vector<std::string> ParseSimple(std::string const& sentence) const;
	std::vector<std::wstring> ParseSimple(std::wstring const& sentence) const;

	//原文の表現のまま + 品詞 <word, word_classs>
	std::vector< std::tuple<std::string, WordClass> > ParseSimpleWithWC(std::string const& sentence) const;
	std::vector< std::tuple<std::wstring, WordClass> > ParseSimpleWithWC(std::wstring const& sentence) const;

	//原形に変換 (skip：原形が存在しないものは無視するか)
	std::vector<std::string> ParseGenkei(std::string const& sentence, bool skip = true) const;
	std::vector<std::wstring> ParseGenkei(std::wstring const& sentence, bool skip = true) const;

	//原形に変換 + 品詞 <word, word_classs> (skip：原形が存在しないものは無視する)
	std::vector< std::tuple<std::string, WordClass> > ParseGenkeiWithWC(std::string const& sentence, bool skip = true) const;
	std::vector< std::tuple<std::wstring, WordClass> > ParseGenkeiWithWC(std::wstring const& sentence, bool skip = true) const;

	//原文の表現のまま (pred：品詞選択．trueを返した場合のみコンテナに格納する)
	std::vector<std::string> ParseSimpleAndFilter(std::string const& sentence, std::function<bool(WordClass)> const& pred) const;
	std::vector<std::wstring> ParseSimpleAndFilter(std::wstring const& sentence, std::function<bool(WordClass)> const& pred) const;

	//原形に変形 (pred：品詞選択．trueを返した場合のみコンテナに格納する)
	std::vector<std::string> ParseGenkeiAndFilter(std::string const& sentence, std::function<bool(WordClass)> const& pred) const;
	std::vector<std::wstring> ParseGenkeiAndFilter(std::wstring const& sentence, std::function<bool(WordClass)> const& pred) const;
};

}

#endif

