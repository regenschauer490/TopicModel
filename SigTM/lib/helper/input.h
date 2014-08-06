/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_INPUT_H
#define SIGTM_INPUT_H

#include "data_format.hpp"
#include "SigUtil/lib/file.hpp"

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
	friend class LDA_CVB0;
	friend class MRInputIterator;

	uint doc_num_;
	TokenList tokens_;		// 単語トークン列
	WordSet words_;			// 単語集合

	std::vector<FilepassString> doc_names_;	// 入力ファイル名
	FilepassString working_directory_;

private:
	InputData() = delete;
	InputData(InputData const& src) = delete;
	InputData(FilepassString folder_pass) : working_directory_(sig::modify_dirpass_tail(folder_pass, true)){ reconstruct(); }

	bool parseLine(std::wstring const& line);
	void reconstruct();
	
protected:
	InputData(uint doc_num, FilepassString working_directory) : doc_num_(doc_num), working_directory_(sig::modify_dirpass_tail(working_directory, true)){};
	void save();

public:
	virtual ~InputData(){}

	// 専用形式の自作データ or 以前の中間出力から読み込む
	// folder_pass: 上記形式のファイルが保存されているディレクトリ
	static InputDataPtr makeInstance(FilepassString folder_pass){ return InputDataPtr(new InputData(folder_pass)); }

	auto getInputFileNames() const->std::vector<FilepassString>{ return doc_names_; }

	uint getDocNum() const{ return doc_num_; }
	uint getWordNum() const{ return words_.size(); }
};

}
#endif