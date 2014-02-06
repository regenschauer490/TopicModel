#ifndef __INPUT_CONTAINER_H__
#define __INPUT_CONTAINER_H__

/*
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <functional>
*/
#include "../sigdm.hpp"

#if USE_MECAB
#include "SigNlp/process_word.h"
#else
#include "SigNlp/signlp.hpp"
#endif

namespace sigdm{

using signlp::WordClass;

struct Token;
typedef std::shared_ptr<Token const> TokenPtr; 
class InputDataFactory;
typedef std::shared_ptr<InputDataFactory> InputDataPtr;


std::function< void(std::wstring&) > const df = [](std::wstring& s){};


/* ����P���\���g�[�N�� */
struct Token {
	uint const self_id;
	uint const doc_id;
	uint const word_id;

	Token(uint self_id, uint document_id, uint unique_word_id) : self_id(self_id), doc_id(document_id), word_id(unique_word_id){}

private:
	Token();
};
	

/* ���̓f�[�^�ւ̃t�B���^�����̐ݒ���s���N���X */
class FilterSetting{
	friend class InputDataFactory;

	bool _base_form;
	std::unordered_set<WordClass> _selected_word_class;
	std::unordered_map< int, std::unordered_set<std::wstring> > _excepted_words;
	std::function< void(std::wstring&) > _pre_filter;
	std::function< void(std::wstring&) > _aft_filter;
		
private:
	FilterSetting();// = delete;

	//_word_class �ɐݒ肳�ꂽ�i���ł��邩
	bool IsSelected_(WordClass self) const{ return _selected_word_class.count(self); }

public:
	//�I�u�W�F�N�g�̐���
	//use_base_form�F�`�ԑf��͌�ɒP������^�ɏC�����邩 (false:���\��, true:���`) 
	FilterSetting(bool use_base_form) : _base_form(use_base_form),  _selected_word_class(), _pre_filter(df), _aft_filter(df){};


	/* �g�[�N�����X�g�ɒǉ�����P��Ɋւ���ݒ� */

	//�`�ԑf��͌�A���X�g�ɒǉ�����i�����w��
	void AddWordClass(WordClass select){ _selected_word_class.insert(select); }

	//�w��h�L�������g���ŏ��O����P����w�� (document_id��0����)
	void AddExceptWord(uint document_id, std::wstring const& word){ _excepted_words[document_id].insert(word); }


	/* ���̓f�[�^�̕�����ɑ΂��čs���t�B���^�����̓o�^ (��F���K�\����URL������)  */

	//�`�ԑf��͑O�ɍs���t�B���^������ݒ�
	void SetPreFilter(std::function< void(std::wstring&) > const& filter){ _pre_filter = filter; }

	//�`�ԑf��͌�ɍs���t�B���^������ݒ�
	void SetAftFilter(std::function< void(std::wstring&) > const& filter){ _aft_filter = filter; }
};


/* ���̓f�[�^������`���֕ϊ�����N���X */
class InputDataFactory {
	friend class LDA;
	friend class TfIdf;

	int _doc_num ;
	std::vector<TokenPtr> _tokens;
	std::vector<C_WStrPtr> _words;
	std::unordered_map<std::wstring, uint const> _word2id_map;

	FilterSetting const _filter;

private:
	InputDataFactory() = delete;
//	InputDataFactory(Document const& document, std::wstring const& save_folder, FilterPtr const& filter) : _doc_num(documents.size()), _filter(filter){ std::vector< std::vector<std::string> > input(1, documents); _MakeData(input, save_folder); }
	InputDataFactory(std::wstring const& folder_pass) : _filter(nullptr){ ReconstructData_(folder_pass); }
	InputDataFactory(InputDataFactory const& src) :_doc_num(src._doc_num),_tokens(src._tokens),_words(src._words),_filter(src._filter){};

	int ParseLine_(std::string const& line, uint& tct);
	void ReconstructData_(std::wstring const& folder_pass);

#if USE_MECAB
	InputDataFactory(Documents const& raw_texts, FilterSetting const& filter, std::wstring const& folder_pass) : _doc_num(raw_texts.size()), _filter(filter){ MakeData_(raw_texts, folder_pass); }
	
	void MakeData_(Documents const& raw_texts, std::wstring const& folder_pass);
#endif
	
public:
	/* �`�ԑf��͑O�̐��̃h�L�������g�Q����͂���ꍇ */

#if USE_MECAB
	//�����̃h�L�������g����� ( wstring raw_texts[document_id][sentence_line] ) 
	static InputDataPtr MakeInstance(Documents const& raw_texts, FilterSetting const& filter, std::wstring const& save_folder_pass){ return InputDataPtr(new InputDataFactory(raw_texts, filter, save_folder_pass)); }
	//�����̃h�L�������g����� ( string raw_texts[document_id][sentence_line] ) 
	static InputDataPtr MakeInstance(std::vector<std::vector<std::string>> const& raw_texts, FilterSetting const& filter, std::wstring const& save_folder_pass){
		Documents tmp(raw_texts.size());
		for(uint i=0; i<raw_texts.size(); ++i) std::transform(raw_texts[i].begin(), raw_texts[i].end(), std::back_inserter(tmp[i]), [](std::string const& s){ return sig::STRtoWSTR(s); });
		return InputDataPtr(new InputDataFactory(tmp, filter, save_folder_pass)); 
	}
#endif
	/* �ȑO�̒��ԏo�͂���͂Ƃ���ꍇ (�t�@�C�����ۑ�����Ă���f�B���N�g�����w��) */
	static InputDataPtr MakeInstance(std::wstring const& folder_pass){ return InputDataPtr(new InputDataFactory(folder_pass)); }
};

}	//namespace sigdm

#endif