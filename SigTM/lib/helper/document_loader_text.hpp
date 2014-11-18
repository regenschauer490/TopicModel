/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_INPUT_FILTER_H
#define SIGTM_INPUT_FILTER_H

#include "../sigtm.hpp"

#if USE_SIGNLP

#include "document_loader.hpp"
#include "SigNlp/polar_evaluation.hpp"
#include "SigUtil/lib/functional/high_order.hpp"
#include "SigUtil/lib/modify/remove.hpp"
#include "SigUtil/lib/calculation/basic_statistics.hpp"
#include <future>

namespace sigtm
{
using signlp::WordClass;


/* 入力データへのフィルタ処理の設定を行う */
class FilterSetting
{
	friend class DocumentLoaderFromText;

	bool base_form_;
	std::unordered_set<WordClass> selected_word_class_;
	std::unordered_map< uint, std::unordered_set<std::wstring> > excepted_words_;
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
	FilterSetting(bool use_base_form) : base_form_(use_base_form), selected_word_class_(), pre_filter_([](std::wstring& s){}), aft_filter_([](std::wstring& s){}){};


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
class DocumentLoaderFromText : public DocumentLoader
{
	const FilterSetting filter_;

private:
	DocumentLoaderFromText() = delete;
	DocumentLoaderFromText(DocumentLoaderFromText const& src) = delete;
	DocumentLoaderFromText(DocumentType type, Documents const& raw_texts, FilterSetting const& filter, FilepassString save_folder_pass, std::vector<FilepassString> const& doc_names)
		: DocumentLoader(type, raw_texts.size(), save_folder_pass), filter_(filter)
	{
		if (doc_names.empty()) for (uint i = 0; i<raw_texts.size(); ++i) info_.doc_names_.push_back(sig::to_fpstring(i));
		else{
			assert(raw_texts.size() == doc_names.size());
			for (auto e : doc_names) info_.doc_names_.push_back(sig::split(e, L".")[0]);
		}

		makeData(type, raw_texts);
		save();
	}
	
	void makeData(DocumentType type, Documents const& raw_texts);
	
public:
	/* 形態素解析前の生のテキストからモデルへの入力形式データを生成する */

	// 変数に保持している一般的なdocument集合から生成 ( raw_texts[document_id][sentence_line] ) 
	static DocumentSetPtr makeInstance(
		Documents const& raw_texts,				// 生のテキストデータ
		FilterSetting const& filter,			// テキストへのフィルタ処理
		FilepassString const& save_folder_pass,		// 作成した入力データの保存先
		std::vector<FilepassString> doc_names		// 各documentの識別名
	){
		return DocumentSetPtr(new DocumentLoaderFromText(DocumentType::Defaut, raw_texts, filter, save_folder_pass, doc_names));
	}

	// テキストファイルに保存された一般的なdocument集合から生成 (各.txtファイルがdocumentに相当)
	static DocumentSetPtr makeInstance(
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
		
		return DocumentSetPtr(new DocumentLoaderFromText(
			DocumentType::Defaut,
			sig::map([&](FilepassString file){
				return sig::str_to_wstr(sig::fromJust(sig::load_line(sig::modify_dirpass_tail(src_folder_pass, true) + file))); 
				}, sig::fromJust(doc_passes)
			), filter, save_folder_pass, doc_names ? sig::fromJust(doc_names) : sig::fromJust(doc_passes))
		);
	}

	// 変数に保持しているtweet集合から生成 ( raw_tweets[user_id][tweet_id] ) 
	static DocumentSetPtr makeInstanceFromTweet(
		Documents const& raw_tweets,			// 生のテキストデータ
		FilterSetting const& filter,			// テキストへのフィルタ処理
		FilepassString const& save_folder_pass,		// 作成した入力データの保存先
		std::vector<FilepassString> user_names		// 各userの識別名
	){
		return DocumentSetPtr(new DocumentLoaderFromText(DocumentType::Tweet, raw_tweets, filter, save_folder_pass, user_names));
	}

	// テキストファイルに保存されたtweet集合から生成 (各.txtファイルがユーザのtweet集合、各行がtweetに相当)
	static DocumentSetPtr makeInstanceFromTweet(
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

		return DocumentSetPtr(new DocumentLoaderFromText(
			DocumentType::Tweet,
			sig::map([&](FilepassString file){
				return sig::str_to_wstr(sig::fromJust(sig::load_line(sig::modify_dirpass_tail(src_folder_pass, true) + file)));
			}, sig::fromJust(doc_passes)
			), filter, save_folder_pass, user_names ? sig::fromJust(user_names) : sig::fromJust(doc_passes))
		);
	}
};


inline void DocumentLoaderFromText::makeData(DocumentType type, Documents const& raw_texts)
{

	const auto ParallelFunc = [](Document document, FilterSetting const& filter){
		std::vector<std::vector<std::wstring>> result;
		auto& mecab = signlp::MecabWrapper::getInstance();

		for (auto& sentence : document){
			filter.pre_filter_(sentence);		//形態素解析前フィルタ処理

			//形態素解析処理
			auto parsed = [&]{
				if (filter.base_form_) return mecab.parseGenkeiThroughFilter(sentence, [&](WordClass wc){ return filter.selected_word_class_.count(wc); });
				else return mecab.parseSimpleThroughFilter(sentence, [&](WordClass wc){ return filter.selected_word_class_.count(wc); });
			}();

			for (auto& word : parsed){
				filter.aft_filter_(word);	//形態素解析後フィルタ処理
			}

			sig::remove_all(parsed, L"");

			result.push_back(std::vector<std::wstring>());
			std::move(parsed.begin(), parsed.end(), std::back_inserter(result.back()));
		}

		return std::move(result);
	};

	std::cout << "make token data : " << std::endl;

	std::vector< std::future< std::vector<std::vector<std::wstring>> > > results;

	for (auto const& document : raw_texts){
		results.push_back(std::async(std::launch::async, ParallelFunc, document, filter_));
	}

	std::vector<std::vector<std::vector<std::wstring>>> doc_line_words;

	for (auto& result : results){
		doc_line_words.push_back(result.get());
	}
	for (uint i = 0; i<doc_line_words.size(); ++i){
		std::wcout << info_.doc_names_[i] << L" parsed. word-num: " << sig::sum(doc_line_words[i], [](std::vector<std::wstring> const& e){ return e.size(); }) << std::endl;
	}
	std::cout << std::endl;

	int token_ct = 0;
	int doc_id = 0;
	for (auto const& line_words : doc_line_words){
		int line_id = 0;
		for (auto const& words : line_words){
			for (auto const& word : words){
				auto wp = std::make_shared<std::wstring const>(word);

				//wordが既出か判定
				if (words_.hasElement(wp)){
					if (DocumentType::Tweet == type) tokens_.push_back(Token(token_ct, doc_id, line_id, words_.getWordID(wp)));
					else tokens_.push_back(Token(token_ct, doc_id, words_.getWordID(wp)));
					++token_ct;
				}
				else{
					uint index = words_.size();
					words_.emplace(index, wp);
					if (DocumentType::Tweet == type) tokens_.push_back(Token(token_ct, doc_id, line_id, index));
					else tokens_.push_back(Token(token_ct, doc_id, index));
					++token_ct;
				}
			}
			++line_id;
		}
		++doc_id;
	}
	info_.is_token_sorted_ = true;

	/*			//指定語彙の除去
	if(!filter_->excepted_words_.empty()){
	filter_->excepted_words_[doc_id].count(

	std::wstring wqstr, tmp;
	const auto& et = filter_->_except_terms[doc_id];

	auto snit = et.begin();
	if(snit != et.end()){
	tmp = sig::STRtoWSTR(*snit);
	wqstr = L"(" + tmp;
	++snit;
	for(auto& snend = et.end(); snit != snend; ++snit){
	tmp = sig::STRtoWSTR(*snit);
	wqstr = (wqstr + L"|" + tmp);
	}
	wqstr = wqstr + L")";

	std::wregex qreg(wqstr);
	std::wstring rep = std::regex_replace(wline, qreg, std::wstring(L""));
	wline = rep;
	}
	}
	*/
}

#endif	// use nlp

}
#endif