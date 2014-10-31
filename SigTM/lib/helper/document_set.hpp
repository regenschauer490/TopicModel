/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_DOCUMENT_SET_H
#define SIGTM_DOCUMENT_SET_H

#include "data_format.hpp"
#include "SigUtil/lib/file/save.hpp"
#include "SigUtil/lib/modify/sort.hpp"

namespace sigtm
{

const std::function< void(std::wstring&) > df = [](std::wstring& s){};

class DocumentSet;
using DocumentSetPtr = std::shared_ptr<DocumentSet const>;

/* 入力する文書の種類 */
enum class DocumentType{ Defaut = 0, Tweet = 1 };


struct DocumentSet
{
/*	friend class LDA_Gibbs;
	friend class MrLDA;
	friend class LDA_CVB0;
	friend class TwitterLDA;
	friend class MRInputIterator;
*/

	DocumentType doc_type_;
	bool is_token_sorted_;	 // priority1: user_id, priority2: doc_id

	uint doc_num_;
	TokenList tokens_;		// 単語トークン列
	WordSet words_;			// 単語集合

	std::vector<FilepassString> doc_names_;	// 入力ファイル名
	FilepassString working_directory_;
	
public:
	DocumentSet() = delete;
	DocumentSet(DocumentSet const& src) = delete;
	DocumentSet(FilepassString folder_pass)
		: working_directory_(sig::modify_dirpass_tail(folder_pass, true)){}
	DocumentSet(DocumentType type, uint doc_num, FilepassString working_directory)
		: doc_type_(type), is_token_sorted_(false), doc_num_(doc_num), working_directory_(sig::modify_dirpass_tail(working_directory, true)){};

	virtual ~DocumentSet(){}

	void save() const;

	void sortToken();

	auto getInputFileNames() const->std::vector<FilepassString>{ return doc_names_; }

	uint getDocNum() const{ return doc_num_; }
	uint getWordNum() const{ return words_.size(); }
};



inline void DocumentSet::sortToken()
{
	sig::sort(tokens_, [](Token const& a, Token const& b){ return a.doc_id < b.doc_id; });
	sig::sort(tokens_, [](Token const& a, Token const& b){ return a.user_id < b.user_id; });
	is_token_sorted_ = true;
}


inline void DocumentSet::save() const
{
	auto base_pass = sig::modify_dirpass_tail(working_directory_, true);

	auto vocab_pass = base_pass + VOCAB_FILENAME;
	auto token_pass = base_pass + TOKEN_FILENAME;

	sig::clear_file(vocab_pass);
	sig::clear_file(token_pass);

	std::locale loc = std::locale("japanese").combine< std::numpunct<char> >(std::locale::classic());
	std::locale::global(loc);

	// save words
	std::wofstream ofs2(vocab_pass);
	for (auto const& word : words_){
		ofs2 << *words_.getWord(word) << std::endl;
	}
	ofs2.close();

	// save tokens
	std::ofstream ofs(token_pass);
	ofs << static_cast<int>(doc_type_) << std::endl;
	ofs << static_cast<int>(is_token_sorted_) << std::endl;
	ofs << doc_num_ << std::endl;
	ofs << words_.size() << std::endl;
	ofs << tokens_.size() << std::endl;

	std::cout << "word_num: " << words_.size() << std::endl << "token_num: " << tokens_.size() << std::endl << std::endl;

	if (DocumentType::Tweet == doc_type_){
		for (auto const& token : tokens_){
			// user doc(tweet) word
			ofs << token.user_id + 1 << " " << token.doc_id + 1 << " " << token.word_id + 1 << std::endl;
		}
	}
	else{
		std::vector< std::unordered_map<uint, uint> > d_w_ct(doc_num_);

		for (auto const& token : tokens_){
			if (d_w_ct[token.doc_id].count(token.word_id)) ++d_w_ct[token.doc_id][token.word_id];
			else d_w_ct[token.doc_id].emplace(token.word_id, 1);
		}

		for (uint d = 0; d < doc_num_; ++d){
			for (auto const& w2ct : d_w_ct[d]){
				// doc word count
				ofs << (d + 1) << " " << (w2ct.first + 1) << " " << w2ct.second << std::endl;
			}
		}
	}

	// save docnames
	sig::save_line(doc_names_, base_pass + DOC_FILENAME);
}

}
#endif