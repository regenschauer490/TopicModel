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
#include "SigUtil/lib/calculation/for_each.hpp"

namespace sigtm
{
class DocumentSet;
using DocumentSetPtr = std::shared_ptr<DocumentSet const>;

/* 入力する文書の種類 */
enum class DocumentType{ Defaut = 0, Tweet = 1 };


struct DocumentLoaderSetInfo
{
	DocumentType doc_type_;			// 入力documentの種類
	bool is_token_sorted_;			// token列のソートの有無．priority1: user_id, priority2: doc_id
	uint doc_num_;					// 入力ファイル数(document数)

	std::vector<FilepassString> doc_names_;	// 入力ファイル名(document名)
	FilepassString working_directory_;		// 入出力先フォルダ
};


class DocumentSet
{
	friend class MRInputIterator;
	friend class LDA_Gibbs;
	friend class LDA_CVB0;
	friend class MrLDA;
	friend class TwitterLDA;
	friend class CTR;

protected:	
	TokenList tokens_;		// 単語トークン列
	WordSet words_;			// 単語集合

	DocumentLoaderSetInfo info_;
	
public:
	DocumentSet(DocumentSet const& src) = delete;
	DocumentSet(FilepassString folder_pass){
		info_.working_directory_ = sig::modify_dirpass_tail(folder_pass, true);
	}
	DocumentSet(DocumentType type, uint doc_num, FilepassString working_directory){
		info_.doc_type_ = type;
		info_.is_token_sorted_ = false;
		info_.doc_num_ = doc_num;
		info_.working_directory_ = sig::modify_dirpass_tail(working_directory, true);
	}

	virtual ~DocumentSet(){}

	void save() const;

	void sortToken();

	auto getDevidedDocument() const->VectorD<std::vector<DocumentId>>;

	auto getInputFileNames() const ->std::vector<FilepassString> const&{ return info_.doc_names_; }
	auto getWorkingDirectory() const->FilepassString{ return info_.working_directory_; }
	auto getDocumentType() const ->DocumentType{ return info_.doc_type_; }
	uint getDocNum() const{ return info_.doc_num_; }
	uint getWordNum() const{ return words_.size(); }
	uint getTokenNum() const{ return tokens_.size(); }
	bool IsTokenSorted() const{ return info_.is_token_sorted_; }
};



inline void DocumentSet::sortToken()
{
	sig::sort(tokens_, [](Token const& a, Token const& b){ return a.doc_id < b.doc_id; });
	
	if(info_.doc_type_ == DocumentType::Tweet){
		sig::sort(tokens_, [](Token const& a, Token const& b){ return a.user_id < b.user_id; });
	}
	
	info_.is_token_sorted_ = true;
}

inline auto DocumentSet::getDevidedDocument() const->VectorD<std::vector<DocumentId>>
{
	VectorD<std::vector<DocumentId>> result(info_.doc_num_);
	
	sig::for_each([&](uint i, Token const& t){
		result[t.doc_id].push_back(i);
	},
	0, tokens_
	);

	return result;
}

/*
inline auto DocumentSet::getDevidedDocument() ->VectorD<std::pair<TokenIter, TokenIter>>
{
VectorD<std::pair<TokenIter, TokenIter>> result(doc_num_);

if (!is_token_sorted_) sortToken();

DocumentId d = 0;
TokenIter begin = tokens_.begin(), end;

for (auto const& t : tokens_){
if (d > t.doc_id){
result[d] = std::make_pair(begin, );
}
result[t.doc_id].push_back();
}
}
*/

inline void DocumentSet::save() const
{
	auto base_pass = sig::modify_dirpass_tail(info_.working_directory_, true);

	auto vocab_pass = base_pass + VOCAB_FILENAME;
	auto token_pass = base_pass + TOKEN_FILENAME;

	sig::clear_file(vocab_pass);
	sig::clear_file(token_pass);
	
	std::vector<Text> tmp;

	// save words
	for (uint i = 0; i < words_.size(); ++i) {
		//sig::save_line(*words_.getWord(word), vocab_pass, sig::WriteMode::append);
		//ofs2 << *words_.getWord(word) << std::endl;
		tmp.push_back(*words_.getWord(i));
	}
	sig::save_line(tmp, vocab_pass);

	auto loc = std::locale("japanese").combine< std::numpunct<char> >(std::locale::classic());
	std::locale::global(loc);

	// save tokens
	std::ofstream ofs(token_pass);
	ofs << static_cast<int>(info_.doc_type_) << std::endl;
	ofs << static_cast<int>(info_.is_token_sorted_) << std::endl;
	ofs << info_.doc_num_ << std::endl;
	ofs << words_.size() << std::endl;
	ofs << tokens_.size() << std::endl;

	std::cout << "word_num: " << words_.size() << std::endl << "token_num: " << tokens_.size() << std::endl << std::endl;

	if (DocumentType::Tweet == info_.doc_type_){
		for (auto const& token : tokens_){
			// user doc(tweet) word
			ofs << token.user_id + 1 << " " << token.doc_id + 1 << " " << token.word_id + 1 << std::endl;
		}
	}
	else{
		std::vector< std::unordered_map<uint, uint> > d_w_ct(info_.doc_num_);

		for (auto const& token : tokens_){
			if (d_w_ct[token.doc_id].count(token.word_id)) ++d_w_ct[token.doc_id][token.word_id];
			else d_w_ct[token.doc_id].emplace(token.word_id, 1);
		}

		for (uint d = 0; d < info_.doc_num_; ++d){
			for (auto const& w2ct : d_w_ct[d]){
				// doc word count
				ofs << (d + 1) << " " << (w2ct.first + 1) << " " << w2ct.second << std::endl;
			}
		}
	}

	// save docnames
	sig::save_line(info_.doc_names_, base_pass + DOC_FILENAME);
}

}
#endif