/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_DOCUMENT_LOADER_HPP
#define SIGTM_DOCUMENT_LOADER_HPP

#include "document_set.hpp"
#include "SigUtil/lib/file/load.hpp"
#include "SigUtil/lib/calculation/for_each.hpp"

namespace sigtm
{

/**
\brief
	モデルへの入力データを作成（専用形式のファイルから読み込み）\n
\detail
	以前に出力されたデータを読み込む．\n
	自然言語で書かれた生のテキストファイルから入力データを作成する場合，日本語には DocumentLoaderFromJapanese ，英語には DocumentLoaderFromEnglish を利用．
*/
class DocumentLoader : public DocumentSet
{
public:
	using DocLineWords = std::vector<std::vector<std::vector<std::wstring>>>;
	using PF = std::function<DocumentLoaderSetInfo(TokenList& tokens, WordSet& words)>;

private:
	DocumentLoader(FilepassString src_directory, FilepassString working_directory)
		: DocumentSet(working_directory){ reconstruct(src_directory); }

	DocumentLoader(PF const& parser, FilepassString working_directory)
		: DocumentSet(working_directory) { info_ = reconstruct(parser); }

	bool parseLine(std::wstring const& line);

	void reconstruct(FilepassString src_directory);
	auto reconstruct(PF const& parser)->DocumentLoaderSetInfo{ return parser(tokens_, words_); }

protected:
	DocumentLoader(DocumentType type, uint doc_num, FilepassString working_directory)
		: DocumentSet(type, doc_num, working_directory){};
	
	auto RemoveMinorWord(DocLineWords& src, uint threshold_num) const->std::unordered_map<Text, uint>;
public:
	virtual ~DocumentLoader(){}

	/**
	\brief
		以前の出力(もしくは自分で用意した) token, vocab ファイルを読み込む
	\detail
		\param src_directory 上記ファイルが保存されているディレクトリ
		\param working_directory 出力データの保存先
		\return モデルへの入力データ
	*/
	static DocumentSetPtr makeInstance(FilepassString src_directory, FilepassString working_directory){
		return DocumentSetPtr(new DocumentLoader(src_directory, working_directory));
	}

	/**
	\brief
		必要な初期化(token,word,DocumentLoaderSetInfo)を手動で行う (非推奨)
	*/
	static DocumentSetPtr makeInstance(PF const& parser, FilepassString working_directory){
		return DocumentSetPtr(new DocumentLoader(parser, working_directory));
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
		const uint tsize = tokens_.size();
		tokens_.push_back(Token(tsize, elem1 - 1, elem2 - 1, elem3 - 1));
	}
	else{
		for (int i = 0; i < elem3; ++i){
			const uint tsize = tokens_.size();
			tokens_.push_back(Token(tsize, elem1 - 1, elem2 - 1));
		}
	}

	return true;
}

namespace impl
{
template <class R>
auto fileopen(FilepassString pass) ->std::vector<R>
{
	auto m_text = sig::load_line<R>(pass);
	if (!isJust(m_text)){
		sig::FileOpenErrorPrint(pass);
		assert(false);
	}
	return fromJust(std::move(m_text));
};
}

inline void DocumentLoader::reconstruct(FilepassString src_directory)
{
	const auto base_pass = sig::modify_dirpass_tail(src_directory, true);

	const auto token_text = impl::fileopen<std::wstring>(base_pass + TOKEN_FILENAME);
	uint line_iter = 0;

	// get feature size
	info_.doc_type_ = static_cast<DocumentType>(std::stoi(token_text[line_iter]));
	info_.is_token_sorted_ = std::stoi(token_text[++line_iter]) > 0 ? true : false;
	const int doc_num = std::stoi(token_text[++line_iter]);
	const int wnum = std::stoi(token_text[++line_iter]);
	const int tnum = std::stoi(token_text[++line_iter]);

	std::cout << "document_num: " << doc_num << std::endl << "word_num:" << wnum << std::endl << "token_num:" << tnum << std::endl << std::endl;

	if (doc_num <= 0 || tnum <= 0 || wnum <= 0) {
		std::cout << "header info is invalid in token file" << std::endl;
		getchar();	std::terminate();
	}

	info_.doc_num_ = static_cast<uint>(doc_num);
	//words_.reserve(wnum);
	tokens_.reserve(tnum);

	for (uint i = ++line_iter; i<token_text.size(); ++i){
		if (!parseLine(token_text[i]) && token_text[i] != L"") {
			std::cout << "error in token file at line : " << i << std::endl;
			getchar();	std::terminate();
		}
	}

	if (tnum != static_cast<int>(tokens_.size())) {
		std::cout << "token file is corrupted" << std::endl;
		getchar();	std::terminate();
	}

	const auto vocab_text = impl::fileopen<std::wstring>(base_pass + VOCAB_FILENAME);

	for (int i = 0; i < wnum; ++i) {
		const auto word = std::make_shared<std::wstring>(vocab_text[i]);

		if (words_.hasElement(word)) {
			words_.emplace(i, word->append(L"_" + std::to_wstring(i)));
			continue;
		}

		words_.emplace(i, word);		
	}
	
	info_.doc_names_ = impl::fileopen<FilepassString>(base_pass + DOC_FILENAME);	

	if (wnum != static_cast<int>(words_.size())) {
		std::cout << "vocab file is corrupted" << std::endl;
		getchar();	std::terminate();
	}
}

inline auto DocumentLoader::RemoveMinorWord(DocLineWords& src, uint threshold_num) const->std::unordered_map<Text, uint>
{
	std::unordered_map<Text, uint> ck;

	if (threshold_num < 1) return ck;

	for (auto const& doc : src) {
		for (auto const& line : doc) {
			for (auto const& word : line) {
				if (ck.count(word)) ++ck[word];
				else ck.emplace(word, 0);
			}
		}
	}

	for (auto& doc : src) {
		for (auto& line : doc) {
			for (auto& word : line) {
				if (ck[word] <= threshold_num) word = L"";
			}
		}
	}

	return ck;
}

}
#endif
