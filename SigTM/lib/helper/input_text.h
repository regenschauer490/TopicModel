/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_INPUT_FILTER_H
#define SIGTM_INPUT_FILTER_H

#include "input.h"
#include "SigNLP/signlp.hpp"

#if USE_SIGNLP
#include "SigNlp/polar_evaluation.hpp"
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
	bool isSelected(WordClass self) const{ return selected_word_class_.count(self) > 0 ? true : false; }

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
	InputDataFromText(DocumentType type, Documents const& raw_texts, FilterSetting const& filter, FilepassString save_folder_pass, std::vector<FilepassString> const& doc_names)
		: InputData(type, raw_texts.size(), save_folder_pass), filter_(filter)
	{
		if (doc_names.empty()) for (uint i = 0; i<raw_texts.size(); ++i) doc_names_.push_back(sig::to_fpstring(i));
		else{
			assert(raw_texts.size() == doc_names.size());
			for (auto e : doc_names) doc_names_.push_back(sig::split(e, L".")[0]);
		}

		makeData(type, raw_texts);
		save();
	}
	
	void makeData(DocumentType type, Documents const& raw_texts);
	
public:
	/* 形態素解析前の生のテキストからモデルへの入力形式データを生成する */

	// 変数に保持している一般的なdocument集合から生成 ( raw_texts[document_id][sentence_line] ) 
	static InputDataPtr makeInstance(
		Documents const& raw_texts,				// 生のテキストデータ
		FilterSetting const& filter,			// テキストへのフィルタ処理
		FilepassString const& save_folder_pass,		// 作成した入力データの保存先
		std::vector<FilepassString> doc_names		// 各documentの識別名
	){
		return InputDataPtr(new InputDataFromText(DocumentType::Defaut, raw_texts, filter, save_folder_pass, doc_names));
	}

	// テキストファイルに保存された一般的なdocument集合から生成 (各.txtファイルがdocumentに相当)
	static InputDataPtr makeInstance(
		FilepassString const& src_folder_pass,		// 生のテキストデータが保存されているフォルダ
		FilterSetting const& filter,				// テキストへのフィルタ処理
		FilepassString const& save_folder_pass,		// 作成した入力データの保存先
		Maybe<std::vector<FilepassString>> doc_names = nothing	// 各documentの識別名(デフォルトはファイル名)
	){
		auto doc_passes = sig::get_file_names(src_folder_pass, false);
		if (!sig::isJust(doc_passes)){
			sig::FileOpenErrorPrint(src_folder_pass);
			assert(false);
		}
		
		return InputDataPtr(new InputDataFromText(
			DocumentType::Defaut,
			sig::map([&](FilepassString file){
				return sig::str_to_wstr(sig::fromJust(sig::load_line(sig::modify_dirpass_tail(src_folder_pass, true) + file))); 
				}, sig::fromJust(doc_passes)
			), filter, save_folder_pass, doc_names ? sig::fromJust(doc_names) : sig::fromJust(doc_passes))
		);
	}

	// 変数に保持しているtweet集合から生成 ( raw_tweets[user_id][tweet_id] ) 
	static InputDataPtr makeInstanceFromTweet(
		Documents const& raw_tweets,			// 生のテキストデータ
		FilterSetting const& filter,			// テキストへのフィルタ処理
		FilepassString const& save_folder_pass,		// 作成した入力データの保存先
		std::vector<FilepassString> user_names		// 各userの識別名
	){
		return InputDataPtr(new InputDataFromText(DocumentType::Tweet, raw_tweets, filter, save_folder_pass, user_names));
	}

	// テキストファイルに保存されたtweet集合から生成 (各.txtファイルがユーザのtweet集合、各行がtweetに相当)
	static InputDataPtr makeInstanceFromTweet(
		FilepassString const& src_folder_pass,		// 生のテキストデータが保存されているフォルダ
		FilterSetting const& filter,				// テキストへのフィルタ処理
		FilepassString const& save_folder_pass,		// 作成した入力データの保存先
		Maybe<std::vector<FilepassString>> user_names = nothing	// 各ユーザの識別名(デフォルトはファイル名)
	){
		auto doc_passes = sig::get_file_names(src_folder_pass, false);
		if (!sig::isJust(doc_passes)){
			sig::FileOpenErrorPrint(src_folder_pass);
			assert(false);
		}

		return InputDataPtr(new InputDataFromText(
			DocumentType::Tweet,
			sig::map([&](FilepassString file){
				return sig::str_to_wstr(sig::fromJust(sig::load_line(sig::modify_dirpass_tail(src_folder_pass, true) + file)));
			}, sig::fromJust(doc_passes)
			), filter, save_folder_pass, user_names ? sig::fromJust(user_names) : sig::fromJust(doc_passes))
		);
	}
};

}
#endif