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
class FilterSetting{
	friend class InputData;

	bool _base_form;
	std::unordered_set<WordClass> _selected_word_class;
	std::unordered_map< int, std::unordered_set<std::wstring> > _excepted_words;
	std::function< void(std::wstring&) > _pre_filter;
	std::function< void(std::wstring&) > _aft_filter;

private:
	FilterSetting();// = delete;

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
	//	InputData(Document const& document, std::wstring const& save_folder, FilterPtr const& filter) : doc_num_(documents.size()), _filter(filter){ std::vector< std::vector<std::string> > input(1, documents); _MakeData(input, save_folder); }
	InputData(std::wstring const& folder_pass) : _filter(nullptr){ reconstruct(folder_pass); }
	InputData(InputData const& src) :doc_num_(src.doc_num_), tokens_(src.tokens_), words_(src.words_), _filter(src._filter){};


#if USE_SIGNLP
	InputData(Documents const& raw_texts, FilterSetting const& filter, std::wstring const& folder_pass) : doc_num_(raw_texts.size()), _filter(filter){ MakeData_(raw_texts, folder_pass); }

	void MakeData_(Documents const& raw_texts, std::wstring const& folder_pass);
#endif

public:
	/* 形態素解析前の生のドキュメント群を入力する場合 */

#if USE_SIGNLP
	//複数のドキュメントを入力 ( wstring raw_texts[document_id][sentence_line] ) 
	static InputDataPtr makeInstance(Documents const& raw_texts, FilterSetting const& filter, std::wstring const& save_folder_pass){ return InputDataPtr(new InputData(raw_texts, filter, save_folder_pass)); }
	//複数のドキュメントを入力 ( string raw_texts[document_id][sentence_line] ) 
	static InputDataPtr makeInstance(std::vector<std::vector<std::string>> const& raw_texts, FilterSetting const& filter, std::wstring const& save_folder_pass){
		Documents tmp(raw_texts.size());
		for (uint i = 0; i<raw_texts.size(); ++i) std::transform(raw_texts[i].begin(), raw_texts[i].end(), std::back_inserter(tmp[i]), [](std::string const& s){ return sig::STRtoWSTR(s); });
		return InputDataPtr(new InputData(tmp, filter, save_folder_pass));
	}
#endif
	/* 以前の中間出力を入力とする場合 (ファイルが保存されているディレクトリを指定) */
	static InputDataPtr makeInstance(std::wstring const& folder_pass){ return InputDataPtr(new InputData(folder_pass)); }
};

}
#endif