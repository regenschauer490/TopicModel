/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_INPUT_H
#define SIGTM_INPUT_H

#include "data_format.hpp"

namespace sigtm
{

const std::function< void(std::wstring&) > df = [](std::wstring& s){};

class InputData;
using InputDataPtr = std::shared_ptr<InputData const>;

/* 各モデルへの入力データを作成 */
class InputData
{
protected:
	friend class LDA_Gibbs;
	friend class MrLDA;
	friend class MRInputIterator;

	uint doc_num_;
	TokenList tokens_;		// 単語トークン列
	WordSet words_;			// 単語集合

	std::vector<FilepassString> doc_names_;	// 入力ファイル名

private:
	InputData() = delete;
	InputData(InputData const& src) = delete;
	InputData(std::wstring const& folder_pass){ reconstruct(folder_pass); }

	bool parseLine(std::wstring const& line);
	void reconstruct(FilepassString folder_pass);


protected:
	InputData(uint doc_num) : doc_num_(doc_num){};
	void save(FilepassString folder_pass);

public:
	virtual ~InputData(){}

	// 指定の形式のデータから読み込む or 以前の中間出力から読み込む
	// folder_pass: ファイルが保存されているディレクトリ
	static InputDataPtr makeInstance(FilepassString folder_pass){ return InputDataPtr(new InputData(folder_pass)); }

	uint getDocNum() const{ return doc_num_; }
	uint getWordNum() const{ return words_.size(); }
};

}
#endif