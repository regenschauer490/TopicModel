/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_INPUT_CONTAINER_H
#define SIG_INPUT_CONTAINER_H

#include "../sigtm.hpp"

namespace sigtm
{
struct Token;
class InputData;

using TokenPtr = std::shared_ptr<Token const>; 
using InputDataPtr = std::shared_ptr<InputData>;


const std::function< void(std::wstring&) > df = [](std::wstring& s){};


/* ある単語を表すトークン */
struct Token
{
	uint const self_id;
	uint const doc_id;
	uint const word_id;

	Token() = delete;
	Token(uint self_id, uint document_id, uint unique_word_id) : self_id(self_id), doc_id(document_id), word_id(unique_word_id){}
};
	

/* 各モデルへの入力データを作成 */
class InputData
{
	friend class LDA;
	friend class TfIdf;

	uint doc_num_;
	std::vector<TokenPtr> tokens_;
	std::vector<C_WStrPtr> words_;
	std::unordered_map<std::wstring, uint> _word2id_map;

private:
	InputData() = delete;
//	InputData(Document const& document, std::wstring const& save_folder, FilterPtr const& filter) : doc_num_(documents.size()), _filter(filter){ std::vector< std::vector<std::string> > input(1, documents); _MakeData(input, save_folder); }
	InputData(std::wstring const& folder_pass) : _filter(nullptr){ reconstruct(folder_pass); }
	InputData(InputData const& src) :doc_num_(src.doc_num_),tokens_(src.tokens_),words_(src.words_),_filter(src._filter){};

	int parseLine(std::string const& line, uint& tct);
	void reconstruct(std::wstring const& folder_pass);
	
public:
	virtual ~InputData(){}

	// 指定の形式のデータから読み込み　以前の中間出力を入力とする場合
	// folder_pass: ファイルが保存されているディレクトリ
	static InputDataPtr makeInstance(std::wstring const& folder_pass){ return InputDataPtr(new InputData(folder_pass)); }
};

}
#endif