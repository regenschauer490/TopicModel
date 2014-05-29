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
	friend class InputDataFromText;

	bool base_form_;
	std::unordered_set<WordClass> selected_word_class_;
	std::unordered_map< int, std::unordered_set<std::wstring> > excepted_words_;
	std::function< void(std::wstring&) > pre_filter_;
	std::function< void(std::wstring&) > aft_filter_;

private:
	FilterSetting() = delete;

	//_word_class �ɐݒ肳�ꂽ�i���ł��邩
	bool isSelected(WordClass self) const{ return selected_word_class_.count(self); }

public:
	FilterSetting(FilterSetting const&) = default;

	//�I�u�W�F�N�g�̐���
	//use_base_form�F�`�ԑf��͌�ɒP������^�ɏC�����邩 (false:���\��, true:���`) 
	FilterSetting(bool use_base_form) : base_form_(use_base_form), selected_word_class_(), pre_filter_(df), aft_filter_(df){};


	/* �g�[�N�����X�g�ɒǉ�����P��Ɋւ���ݒ� */

	//�`�ԑf��͌�A���X�g�ɒǉ�����i�����w��
	void addWordClass(WordClass select){ selected_word_class_.insert(select); }

	//�w��h�L�������g���ŏ��O����P����w�� (document_id��0����)
	void addExceptWord(uint document_id, std::wstring const& word){ excepted_words_[document_id].insert(word); }


	/* ���̓f�[�^�̕�����ɑ΂��čs���t�B���^�����̓o�^ (��F���K�\����URL������)  */

	//�`�ԑf��͑O�ɍs���t�B���^������ݒ�
	void setPreFilter(std::function< void(std::wstring&) > const& filter){ pre_filter_ = filter; }

	//�`�ԑf��͌�ɍs���t�B���^������ݒ�
	void setAftFilter(std::function< void(std::wstring&) > const& filter){ aft_filter_ = filter; }
};


/* ���R����̃e�L�X�g������̓f�[�^���쐬 */
class InputDataFromText : public InputData
{
	const FilterSetting filter_;

private:
	InputDataFromText() = delete;
	InputDataFromText(InputDataFromText const& src) = delete;
	InputDataFromText(Documents const& raw_texts, FilterSetting const& filter, std::wstring save_folder_pass) : InputData(raw_texts.size()), filter_(filter)
	{
		makeData(raw_texts);
		save(save_folder_pass);
	}
	
	void makeData(Documents const& raw_texts);
	
public:
	// �`�ԑf��͑O�̐��̃e�L�X�g������̓f�[�^�𐶐����� ( raw_texts[document_id][sentence_line] ) 
	static InputDataPtr makeInstance(Documents const& raw_texts, FilterSetting const& filter, std::wstring const& save_folder_pass){
		return InputDataPtr(new InputDataFromText(raw_texts, filter, save_folder_pass)); 
	}
};

}
#endif