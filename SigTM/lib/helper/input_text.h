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
class FilterSetting
{
	friend class InputData;

	bool _base_form;
	std::unordered_set<WordClass> _selected_word_class;
	std::unordered_map< int, std::unordered_set<std::wstring> > _excepted_words;
	std::function< void(std::wstring&) > _pre_filter;
	std::function< void(std::wstring&) > _aft_filter;

private:
	FilterSetting() = delete;

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
	InputDataFromText(Documents const& raw_texts, FilterSetting const& filter, std::wstring folder_pass) 
		: InputData(raw_texts.size()), _filter(filter){ makeData(raw_texts, folder_pass); }
	InputDataFromText(InputDataFromText const& src) = delete;
	
	void makeData(Documents const& raw_texts, std::wstring folder_pass);
	
public:
	// �`�ԑf��͑O�̐��̃e�L�X�g������̓f�[�^�𐶐����� ( raw_texts[document_id][sentence_line] ) 
	static InputDataPtr makeInstance(Documents const& raw_texts, FilterSetting const& filter, std::wstring const& save_folder_pass){
		return InputDataPtr(new InputDataFromText(raw_texts, filter, save_folder_pass)); 
	}
};

}
#endif