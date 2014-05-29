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


/* 入力データへのフィルタ処理の設定を行う */
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

	//_word_class に設定された品詞であるか
	bool isSelected(WordClass self) const{ return selected_word_class_.count(self); }

public:
	FilterSetting(FilterSetting const&) = default;

	//オブジェクトの生成
	//use_base_form：形態素解析後に単語を原型に修正するか (false:元表現, true:原形) 
	FilterSetting(bool use_base_form) : base_form_(use_base_form), selected_word_class_(), pre_filter_(df), aft_filter_(df){};


	/* トークンリストに追加する単語に関する設定 */

	//形態素解析後、リストに追加する品詞を指定
	void addWordClass(WordClass select){ selected_word_class_.insert(select); }

	//指定ドキュメント内で除外する単語を指定 (document_idは0から)
	void addExceptWord(uint document_id, std::wstring const& word){ excepted_words_[document_id].insert(word); }


	/* 入力データの文字列に対して行うフィルタ処理の登録 (例：正規表現でURLを除去)  */

	//形態素解析前に行うフィルタ処理を設定
	void setPreFilter(std::function< void(std::wstring&) > const& filter){ pre_filter_ = filter; }

	//形態素解析後に行うフィルタ処理を設定
	void setAftFilter(std::function< void(std::wstring&) > const& filter){ aft_filter_ = filter; }
};


/* 自然言語のテキストから入力データを作成 */
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
	// 形態素解析前の生のテキストから入力データを生成する ( raw_texts[document_id][sentence_line] ) 
	static InputDataPtr makeInstance(Documents const& raw_texts, FilterSetting const& filter, std::wstring const& save_folder_pass){
		return InputDataPtr(new InputDataFromText(raw_texts, filter, save_folder_pass)); 
	}
};

}
#endif