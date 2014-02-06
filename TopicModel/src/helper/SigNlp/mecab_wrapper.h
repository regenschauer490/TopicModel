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

/* MeCab ���b�p */
class MecabWrapper
{
	MeCab::Model* _model;

	//ex)����	����,���,*,*,*,*,����,���E,���[,,
	//   ����	�`�e��,����,*,*,�`�e���E�A�E�I�i,��{�`,����,�l���C,�l���C,�˂ނ�/����,

	//MeCab::Tagger* _tagger_o;	//ex)���� ����
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

	//�����̕\���̂܂�
	std::vector<std::string> ParseSimple(std::string const& sentence) const;
	std::vector<std::wstring> ParseSimple(std::wstring const& sentence) const;

	//�����̕\���̂܂� + �i�� <word, word_classs>
	std::vector< std::tuple<std::string, WordClass> > ParseSimpleWithWC(std::string const& sentence) const;
	std::vector< std::tuple<std::wstring, WordClass> > ParseSimpleWithWC(std::wstring const& sentence) const;

	//���`�ɕϊ� (skip�F���`�����݂��Ȃ����͖̂������邩)
	std::vector<std::string> ParseGenkei(std::string const& sentence, bool skip = true) const;
	std::vector<std::wstring> ParseGenkei(std::wstring const& sentence, bool skip = true) const;

	//���`�ɕϊ� + �i�� <word, word_classs> (skip�F���`�����݂��Ȃ����͖̂�������)
	std::vector< std::tuple<std::string, WordClass> > ParseGenkeiWithWC(std::string const& sentence, bool skip = true) const;
	std::vector< std::tuple<std::wstring, WordClass> > ParseGenkeiWithWC(std::wstring const& sentence, bool skip = true) const;

	//�����̕\���̂܂� (pred�F�i���I���Dtrue��Ԃ����ꍇ�̂݃R���e�i�Ɋi�[����)
	std::vector<std::string> ParseSimpleAndFilter(std::string const& sentence, std::function<bool(WordClass)> const& pred) const;
	std::vector<std::wstring> ParseSimpleAndFilter(std::wstring const& sentence, std::function<bool(WordClass)> const& pred) const;

	//���`�ɕό` (pred�F�i���I���Dtrue��Ԃ����ꍇ�̂݃R���e�i�Ɋi�[����)
	std::vector<std::string> ParseGenkeiAndFilter(std::string const& sentence, std::function<bool(WordClass)> const& pred) const;
	std::vector<std::wstring> ParseGenkeiAndFilter(std::wstring const& sentence, std::function<bool(WordClass)> const& pred) const;
};

}

#endif

