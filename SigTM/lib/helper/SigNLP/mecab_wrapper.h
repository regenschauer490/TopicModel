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

/* MeCab ���b�p */
class MecabWrapper
{
	MeCab::Model* model_;

	//ex)����	����,���,*,*,*,*,����,���E,���[,,
	//   ����	�`�e��,����,*,*,�`�e���E�A�E�I�i,��{�`,����,�l���C,�l���C,�˂ނ�/����,

	//MeCab::Tagger* _tagger_o;	//ex)���� ����
	//_tagger_o(MeCab::createTagger("-Owakati"))

	using TaggerPtr = std::shared_ptr<MeCab::Tagger>;
	using LatticePtr = std::shared_ptr<MeCab::Lattice>;

private:
	MecabWrapper() : model_(MeCab::createModel("")){}
	MecabWrapper(MecabWrapper const&) = delete;

	// ���񏈗����l��
	volatile void ParseImpl(std::string const& src, std::string& dest) const;

public:
	static MecabWrapper& getInstance(){
		static MecabWrapper instance;	//thread safe in C++11
		return instance;
	}

	//�����̕\���̂܂�
	std::vector<std::string> parseSimple(std::string const& sentence) const;
	std::vector<std::wstring> parseSimple(std::wstring const& sentence) const;

	//�����̕\���̂܂� + �i�� <word, word_classs>
	std::vector< std::tuple<std::string, WordClass> > parseSimpleWithWC(std::string const& sentence) const;
	std::vector< std::tuple<std::wstring, WordClass> > parseSimpleWithWC(std::wstring const& sentence) const;

	//���`�ɕϊ� (skip�F���`�����݂��Ȃ����͖̂������邩)
	std::vector<std::string> parseGenkei(std::string const& sentence, bool skip = true) const;
	std::vector<std::wstring> parseGenkei(std::wstring const& sentence, bool skip = true) const;

	//���`�ɕϊ� + �i�� <word, word_classs> (skip�F���`�����݂��Ȃ����͖̂�������)
	std::vector< std::tuple<std::string, WordClass> > parseGenkeiWithWC(std::string const& sentence, bool skip = true) const;
	std::vector< std::tuple<std::wstring, WordClass> > parseGenkeiWithWC(std::wstring const& sentence, bool skip = true) const;

	//�����̕\���̂܂� (pred�F�i���I���Dtrue��Ԃ����ꍇ�̂݃R���e�i�Ɋi�[����)
	std::vector<std::string> parseSimpleThroughFilter(std::string const& sentence, std::function<bool(WordClass)> const& pred) const;
	std::vector<std::wstring> parseSimpleThroughFilter(std::wstring const& sentence, std::function<bool(WordClass)> const& pred) const;

	//���`�ɕό` (pred�F�i���I���Dtrue��Ԃ����ꍇ�̂݃R���e�i�Ɋi�[����)
	std::vector<std::string> parseGenkeiThroughFilter(std::string const& sentence, std::function<bool(WordClass)> const& pred) const;
	std::vector<std::wstring> parseGenkeiThroughFilter(std::wstring const& sentence, std::function<bool(WordClass)> const& pred) const;
};

}
#endif

