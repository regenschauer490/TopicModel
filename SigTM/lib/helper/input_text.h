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
	friend class InputData;

	bool _base_form;
	std::unordered_set<WordClass> _selected_word_class;
	std::unordered_map< int, std::unordered_set<std::wstring> > _excepted_words;
	std::function< void(std::wstring&) > _pre_filter;
	std::function< void(std::wstring&) > _aft_filter;

private:
	FilterSetting() = delete;

	//_word_class に設定された品詞であるか
	bool IsSelected_(WordClass self) const{ return _selected_word_class.count(self); }

public:
	//オブジェクトの生成
	//use_base_form：形態素解析後に単語を原型に修正するか (false:元表現, true:原形) 
	FilterSetting(bool use_base_form) : _base_form(use_base_form), _selected_word_class(), _pre_filter(df), _aft_filter(df){};


	/* トークンリストに追加する単語に関する設定 */

	//形態素解析後、リストに追加する品詞を指定
	void AddWordClass(WordClass select){ _selected_word_class.insert(select); }

	//指定ドキュメント内で除外する単語を指定 (document_idは0から)
	void AddExceptWord(uint document_id, std::wstring const& word){ _excepted_words[document_id].insert(word); }


	/* 入力データの文字列に対して行うフィルタ処理の登録 (例：正規表現でURLを除去)  */

	//形態素解析前に行うフィルタ処理を設定
	void SetPreFilter(std::function< void(std::wstring&) > const& filter){ _pre_filter = filter; }

	//形態素解析後に行うフィルタ処理を設定
	void SetAftFilter(std::function< void(std::wstring&) > const& filter){ _aft_filter = filter; }
};


/* 自然言語のテキストから入力データを作成 */
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
	// 形態素解析前の生のテキストから入力データを生成する ( raw_texts[document_id][sentence_line] ) 
	static InputDataPtr makeInstance(Documents const& raw_texts, FilterSetting const& filter, std::wstring const& save_folder_pass){
		return InputDataPtr(new InputDataFromText(raw_texts, filter, save_folder_pass)); 
	}
};

}
#endif