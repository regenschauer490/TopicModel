/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_INPUT_FILTER_H
#define SIG_INPUT_FILTER_H

#include "input.h"

#if USE_SIGNLP
#include "SigNlp/process_word.h"
#else
#include "SigNLP/signlp.hpp"
#endif

namespace sigtm
{
using signlp::WordClass;


/* ���̓f�[�^�ւ̃t�B���^�����̐ݒ���s�� */
class FilterSetting{
	friend class InputData;

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
	FilterSetting(bool use_base_form) : _base_form(use_base_form), _selected_word_class(), _pre_filter(df), _aft_filter(df){};


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


/* ���R����̃e�L�X�g������̓f�[�^���쐬 */
class InputDataFromText : public InputData
{
	FilterSetting const _filter;

private:
	InputDataFromText() = delete;
	//	InputData(Document const& document, std::wstring const& save_folder, FilterPtr const& filter) : doc_num_(documents.size()), _filter(filter){ std::vector< std::vector<std::string> > input(1, documents); _MakeData(input, save_folder); }
	InputData(std::wstring const& folder_pass) : _filter(nullptr){ reconstruct(folder_pass); }
	InputData(InputData const& src) :doc_num_(src.doc_num_), tokens_(src.tokens_), words_(src.words_), _filter(src._filter){};


#if USE_SIGNLP
	InputData(Documents const& raw_texts, FilterSetting const& filter, std::wstring const& folder_pass) : doc_num_(raw_texts.size()), _filter(filter){ MakeData_(raw_texts, folder_pass); }

	void MakeData_(Documents const& raw_texts, std::wstring const& folder_pass);
#endif

public:
	/* �`�ԑf��͑O�̐��̃h�L�������g�Q����͂���ꍇ */

#if USE_SIGNLP
	//�����̃h�L�������g����� ( wstring raw_texts[document_id][sentence_line] ) 
	static InputDataPtr makeInstance(Documents const& raw_texts, FilterSetting const& filter, std::wstring const& save_folder_pass){ return InputDataPtr(new InputData(raw_texts, filter, save_folder_pass)); }
	//�����̃h�L�������g����� ( string raw_texts[document_id][sentence_line] ) 
	static InputDataPtr makeInstance(std::vector<std::vector<std::string>> const& raw_texts, FilterSetting const& filter, std::wstring const& save_folder_pass){
		Documents tmp(raw_texts.size());
		for (uint i = 0; i<raw_texts.size(); ++i) std::transform(raw_texts[i].begin(), raw_texts[i].end(), std::back_inserter(tmp[i]), [](std::string const& s){ return sig::STRtoWSTR(s); });
		return InputDataPtr(new InputData(tmp, filter, save_folder_pass));
	}
#endif
	/* �ȑO�̒��ԏo�͂���͂Ƃ���ꍇ (�t�@�C�����ۑ�����Ă���f�B���N�g�����w��) */
	static InputDataPtr makeInstance(std::wstring const& folder_pass){ return InputDataPtr(new InputData(folder_pass)); }
};

}
#endif