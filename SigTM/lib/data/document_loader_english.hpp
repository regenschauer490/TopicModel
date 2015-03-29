﻿/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_DOCUMENT_LOADER_ENGLISH_HPP
#define SIGTM_DOCUMENT_LOADER_ENGLISH_HPP

#include "../sigtm.hpp"
#include "document_loader.hpp"
#include "SigUtil/lib/functional/high_order.hpp"
#include "SigUtil/lib/modify/remove.hpp"
#include "SigUtil/lib/calculation/basic_statistics.hpp"
#include <future>

#if SIG_USE_SIGNLP
#include "SigNLP/tree_tagger_wrapper.hpp"
#endif

namespace sigtm
{
#if SIG_USE_SIGNLP
using signlp::WordClass;
#endif

/**
\brief
	モデルへの入力データを作成（英語で書かれたファイルから読み込み）\n
\detail
	英語で書かれた生のテキストファイルから入力データを作成する場合に利用．\n
	通常，1ファイルが1文書に対応している．
	他の文書形式（例：tweet）の場合，適切なファクトリ関数を選択する必要があることに注意．
	
	\sa DocumentLoader
	\sa DocumentLoaderFromJapanese
*/
class DocumentLoaderFromEnglish : public DocumentLoader
{
public:
	/**
	\brief
		入力データへのフィルタ処理の設定を行う
	\detail
		形態素解析の前後の処理，および形態素解析に関する設定を行う．	\n
		SigNLPを使う（sigtm.hpp の SIG_USE_SIGNLP を1に設定する）場合には，形態素解析器 TreeTagger を利用可能（TreeTaggerは別途インストールする必要あり）．
		SigNLPを使わない場合，形態素解析は行わず，単語の分離操作を関数として設定する（defaultは半角スペース区切りで分離）．

		設定項目一覧（※はSigNLPが必要）
		- 単語を「基本形」か「原形」のどちらで扱うか ※
		- 指定回数以下の出現数の単語を除外する
		- 
	*/
	class FilterSetting
	{
	public:
		using Filter = std::function< void(Text&) >;
		using Parser = std::function<std::vector<Text>(Text const&)>;

	private:
		const bool base_form_;
		uint remove_word_count_;
		const FilepassString exe_pass_;
		const FilepassString param_pass_;
		Maybe<Parser> parser_;
		Filter common_pri_filter_;
		Filter common_post_filter_;
		std::unordered_map< uint, Filter> individual_pri_filter_;
		std::unordered_map< uint, Filter> individual_post_filter_;
	
	public:
		FilterSetting(Maybe<Parser> parser = Nothing<Parser>()) : base_form_(false), parser_(parser), common_pri_filter_(nullptr), common_post_filter_(nullptr){};
#if SIG_USE_SIGNLP
		FilterSetting(FilepassString exe_pass, FilepassString param_pass, bool use_base_form)
			: base_form_(use_base_form), exe_pass_(exe_pass), param_pass_(param_pass), common_pri_filter_(nullptr), common_post_filter_(nullptr) {};
#endif
		FilterSetting(FilterSetting const&) = default;

		bool isBaseForm() const { return base_form_; }

		FilepassString getTreeTaggerPass() const { return exe_pass_; }
		FilepassString getTreeTaggerParamPass() const { return param_pass_; }

		Maybe<Parser> getParser() const{ return parser_; }

		// 出現数が指定数以下の単語を除外
		void setRemoveWordCount(uint threshold_num) { remove_word_count_ = threshold_num; }
		uint getRemoveWordCount() const { return remove_word_count_; }

		/* 入力データの文字列に対して行うフィルタ処理の登録 (例：正規表現でURLを除去)  */

		// 各ドキュメントの各行に行うフィルタ処理を設定
		void setCommonPriorFilter(Filter filter){ common_pri_filter_ = filter; }
		void setCommonPosteriorFilter(Filter filter){ common_post_filter_ = filter; }

		Filter getCommonPriorFilter() const{ return common_pri_filter_; }
		Filter getCommonPosteriorFilter() const{ return common_post_filter_; }

		// 指定ドキュメントの各行に行うフィルタ処理を設定 (document_idは0から)
		void setIndividualPriorFilter(uint document_id, Filter filter){ individual_pri_filter_.emplace(document_id, filter); }
		void setIndividualPosteriorFilter(uint document_id, Filter filter){ individual_post_filter_.emplace(document_id, filter); }

		Filter getIndividualPriorFilter(uint document_id) const{ return individual_pri_filter_.count(document_id) ? individual_pri_filter_.at(document_id) : nullptr; }
		Filter getIndividualPosteriorFilter(uint document_id) const{ return individual_post_filter_.count(document_id) ? individual_post_filter_.at(document_id) : nullptr; }
	};

private:
	const FilterSetting filter_;

private:
	DocumentLoaderFromEnglish() = delete;
	DocumentLoaderFromEnglish(DocumentLoaderFromEnglish const& src) = delete;
	DocumentLoaderFromEnglish(DocumentType type, Documents const& raw_texts, FilterSetting filter, FilepassString working_directory, std::vector<FilepassString> const& doc_names)
		: DocumentLoader(type, raw_texts.size(), working_directory), filter_(filter)
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
	/* 生テキストからモデルへの入力形式データを生成する */

	// 変数に保持している一般的なdocument集合から生成 ( raw_texts[document_id][sentence_line] ) 
	static DocumentSetPtr makeInstance(
		Documents const& raw_texts,				// 生のテキストデータ
		FilterSetting filter,					// テキストへのフィルタ処理
		FilepassString working_directory,		// 出力データの保存先
		std::vector<FilepassString> doc_names	// 各documentの識別名
	){
		return DocumentSetPtr(new DocumentLoaderFromEnglish(DocumentType::Defaut, raw_texts, filter, working_directory, doc_names));
	}

	// テキストファイルに保存された一般的なdocument集合から生成 (各.txtファイルがdocumentに相当)
	static DocumentSetPtr makeInstance(
		FilepassString src_folder_pass,			// 生のテキストデータが保存されているフォルダ
		FilterSetting filter,					// テキストへのフィルタ処理
		FilepassString working_directory,		// 出力データの保存先
		Maybe<std::vector<FilepassString>> doc_names = nothing	// 各documentの識別名(デフォルトはファイル名)
	){
		auto doc_passes = sig::get_file_names(src_folder_pass, false);
		if (!isJust(doc_passes)){
			sig::FileOpenErrorPrint(src_folder_pass);
			assert(false);
		}
		
		return DocumentSetPtr(new DocumentLoaderFromEnglish(
			DocumentType::Defaut,
			sig::map([&](FilepassString file){
				return sig::str_to_wstr(fromJust(sig::load_line(sig::modify_dirpass_tail(src_folder_pass, true) + file))); 
				}, fromJust(doc_passes)
			), filter, working_directory, doc_names ? fromJust(doc_names) : fromJust(doc_passes))
		);
	}

	// 変数に保持しているtweet集合から生成 ( raw_tweets[user_id][tweet_id] ) 
	static DocumentSetPtr makeInstanceFromTweet(
		Documents const& raw_tweets,			// 生のテキストデータ
		FilterSetting filter,					// テキストへのフィルタ処理
		FilepassString working_directory,		// 出力データの保存先
		std::vector<FilepassString> user_names	// 各userの識別名
	){
		return DocumentSetPtr(new DocumentLoaderFromEnglish(DocumentType::Tweet, raw_tweets, filter, working_directory, user_names));
	}

	// テキストファイルに保存されたtweet集合から生成 (各.txtファイルがユーザのtweet集合、各行がtweetに相当)
	static DocumentSetPtr makeInstanceFromTweet(
		FilepassString src_folder_pass,			// 生のテキストデータが保存されているフォルダ
		FilterSetting filter,					// テキストへのフィルタ処理
		FilepassString working_directory,		// 出力データの保存先
		Maybe<std::vector<FilepassString>> user_names = nothing	// 各ユーザの識別名(デフォルトはファイル名)
	){
		auto doc_passes = sig::get_file_names(src_folder_pass, false);
		if (!isJust(doc_passes)){
			sig::FileOpenErrorPrint(src_folder_pass);
			assert(false);
		}

		return DocumentSetPtr(new DocumentLoaderFromEnglish(
			DocumentType::Tweet,
			sig::map([&](FilepassString file){
				return sig::str_to_wstr(fromJust(sig::load_line(sig::modify_dirpass_tail(src_folder_pass, true) + file)));
			}, fromJust(doc_passes)
			), filter, working_directory, user_names ? fromJust(user_names) : fromJust(doc_passes))
		);
	}
};


inline void DocumentLoaderFromEnglish::makeData(DocumentType type, Documents const& raw_texts)
{
	const auto ParallelFunc = [](uint id, Document document, FilterSetting const& filter)
	{
		std::vector<std::vector<std::wstring>> result;

#if SIG_USE_SIGNLP
		auto& tagger = signlp::TreeTaggerWrapper::getInstance(filter.getTreeTaggerPass(), filter.getTreeTaggerParamPass());
		const auto parser =  [&](Text const& text) { return filter.isBaseForm() ? tagger.parseGenkei(text) : sig::split(text, L" "); };
#else
		const auto tp = filter.getParser();
		const auto parser = tp ? *tp : [](Text const& text) { return sig::split(text, L" "); };
#endif
		for (auto& sentence : document){
			if(auto f = filter.getCommonPriorFilter()) f(sentence);
			if(auto f = filter.getIndividualPriorFilter(id)) f(sentence);

			auto parsed = parser(sentence);

			for (auto& word : parsed){
				if(auto f = filter.getCommonPosteriorFilter()) f(word);
				if(auto f = filter.getIndividualPosteriorFilter(id)) f(word);
			}

			sig::remove_all(parsed, L"");

			result.push_back(std::vector<std::wstring>());
			std::move(parsed.begin(), parsed.end(), std::back_inserter(result.back()));
		}

		return std::move(result);
	};

	std::cout << "make token data : " << std::endl;

	std::vector<std::vector<std::vector<std::wstring>>> doc_line_words;

	const uint block = 200;
	for (uint d = 0, ds = 1 + raw_texts.size() / block; d < ds; ++d) {
		std::vector< std::future< std::vector<std::vector<std::wstring>> > > results;

		for (uint i = block * d, is = sig::min(block * (d+1), raw_texts.size()); i < is; ++i) {
			results.push_back(std::async(std::launch::async, ParallelFunc, i, raw_texts[i], filter_));
		}
		for (auto& result : results) {
			doc_line_words.push_back(result.get());
		}
	}

	for (uint i = 0; i < doc_line_words.size(); ++i){
		std::wcout << info_.doc_names_[i] << L" parsed. word-num: " << sig::sum(doc_line_words[i], [](std::vector<std::wstring> const& e){ return e.size(); }) << std::endl;
	}
	std::cout << std::endl;

	RemoveMinorWord(doc_line_words, filter_.getRemoveWordCount());

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
}

}
#endif