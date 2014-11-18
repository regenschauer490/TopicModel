/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_INPUT_H
#define SIGTM_INPUT_H

#include "document_set.hpp"
#include "SigUtil/lib/file/load.hpp"
#include "SigUtil/lib/calculation/for_each.hpp"

namespace sigtm
{

/* 各モデルへの入力データを作成 */
class DocumentLoader : public DocumentSet
{
public:
	using PF = std::function<DocumentLoaderSetInfo(TokenList& tokens, WordSet& words)>;

private:
	DocumentLoader(FilepassString folder_pass)
		: DocumentSet(folder_pass){ reconstruct(); }

	DocumentLoader(PF const& parser){ info_ = reconstruct(parser); }

	bool parseLine(std::wstring const& line);

	void reconstruct();
	auto reconstruct(PF const& parser)->DocumentLoaderSetInfo
	{ return parser(tokens_, words_); }

protected:
	DocumentLoader(DocumentType type, uint doc_num, FilepassString working_directory)
		: DocumentSet(type, doc_num, working_directory){};
	
public:
	virtual ~DocumentLoader(){}

	// 専用形式の自作データ or 以前の中間出力から読み込む
	// folder_pass: 上記形式のファイルが保存されているディレクトリ
	static DocumentSetPtr makeInstance(FilepassString folder_pass){
		return DocumentSetPtr(new DocumentLoader(folder_pass)); 
	}

	static DocumentSetPtr makeInstance(PF const& parser){
		return DocumentSetPtr(new DocumentLoader(parser)); 
	}
};


inline bool DocumentLoader::parseLine(std::wstring const& line)
{
	std::wistringstream is(line);
	int elem1 = 0;
	int elem2 = 0;
	int elem3 = 0;

	is >> elem1 >> elem2 >> elem3;
	if (!elem1 || !elem2 || !elem3) {
		std::cout << "parse error" << std::endl;
		return false;
	}

	if (DocumentType::Tweet == info_.doc_type_){
		uint tsize = tokens_.size();
		tokens_.push_back(Token(tsize, elem1 - 1, elem2 - 1, elem3 - 1));
	}
	else{
		for (int i = 0; i < elem3; ++i){
			uint tsize = tokens_.size();
			tokens_.push_back(Token(tsize, elem1 - 1, elem2 - 1));
		}
	}

	return true;
}

inline void DocumentLoader::reconstruct()
{
	auto fileopen = [&](FilepassString pass) ->std::vector<std::wstring>
	{
		auto m_text = sig::load_line<std::wstring>(pass);
		if (!sig::isJust(m_text)){
			sig::FileOpenErrorPrint(pass);
			assert(false);
		}
		return sig::fromJust(std::move(m_text));
	};

	auto base_pass = sig::modify_dirpass_tail(info_.working_directory_, true);

	auto token_text = fileopen(base_pass + TOKEN_FILENAME);
	uint line_iter = 0;

	// get feature size
	info_.doc_type_ = static_cast<DocumentType>(std::stoi(token_text[line_iter]));
	info_.is_token_sorted_ = std::stoi(token_text[++line_iter]) > 0 ? true : false;
	int doc_num = std::stoi(token_text[++line_iter]);
	int wnum = std::stoi(token_text[++line_iter]);
	int tnum = std::stoi(token_text[++line_iter]);

	std::cout << "document_num: " << doc_num << std::endl << "word_num:" << wnum << std::endl << "token_num:" << tnum << std::endl << std::endl;

	if (doc_num <= 0 || tnum <= 0 || wnum <= 0) {
		std::cout << "token file is corrupted" << std::endl;
		assert(false);
	}

	info_.doc_num_ = static_cast<uint>(doc_num);
	//words_.reserve(wnum);
	tokens_.reserve(tnum);

	for (uint i = ++line_iter; i<token_text.size(); ++i){
		if (!parseLine(token_text[i]) && token_text[i] != L"") {
			std::cout << "error in token file at line : " << i << std::endl;
			assert(false);
		}
	}

	auto vocab_text = fileopen(base_pass + VOCAB_FILENAME);

	sig::for_each([&](uint i, std::wstring const& e){
		auto word = std::make_shared<std::wstring>(e);
		words_.emplace(i, word);
	}
	, 0, vocab_text);


	info_.doc_names_ = fileopen(base_pass + DOC_FILENAME);	
}

}
#endif