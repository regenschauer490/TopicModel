/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#include <future>
#include <sstream>
#include "SigUtil/lib/file.hpp"
#include "SigUtil/lib/iteration.hpp"
#include "input.h"

namespace sigtm
{
// 
inline bool InputData::parseLine(std::wstring const& line)
{
	std::wistringstream is(line);
	int doc_id = 0;
	int word_id = 0;
	int count = 0;

	is >> doc_id >> word_id >> count;
	if (!doc_id || !word_id || !count) {
		std::cout << "parse error" << std::endl;
		return false;
	}
  
	for(uint i = 0; i < count; ++i){
		uint tsize = tokens_.size();
		tokens_.push_back( std::make_shared<Token>(tsize, doc_id-1, word_id-1) );
	}

	return true;
}

void InputData::reconstruct(FilepassString folder_pass)
{
	/*
	auto filenames = sig::get_file_names(folder_pass, false);
	if(sig::is_container_valid(filenames)){
		std::wcout << L"folder not found : " << folder_pass << std::endl;
		assert(false);
	}*/

	auto fileopen = [&](FilepassString pass) ->std::vector<std::wstring>
	{	
		auto m_text = sig::read_line<std::wstring>(pass);
		if (!sig::is_container_valid(m_text)){
			sig::FileOpenErrorPrint(pass);
			assert(false);
		}
		return sig::fromJust(std::move(m_text));
	};

	auto base_pass = sig::impl::modify_dirpass_tail(folder_pass, true);

	auto token_text = fileopen(base_pass + TOKEN_FILENAME);

	// get feature size
	int doc_num = std::stoi(token_text[0]);
	int wnum = std::stoi(token_text[1]);
	int tnum = std::stoi(token_text[2]);

	std::cout << "document_num: " << doc_num << std::endl << "word_num:" << wnum << std::endl << "token_num:" << tnum << std::endl;

	if (doc_num <= 0 || tnum <= 0 || wnum <= 0) {
		std::cout << "token file is corrupted" << std::endl;
		assert(false);
	}

	doc_num_ = static_cast<uint>(doc_num);
	words_.reserve(wnum);
	tokens_.reserve(tnum);

	for(uint i=3; i<token_text.size(); ++i){
		if (!parseLine(token_text[i]) && token_text[i] != L"") {
			std::cout << "error in token file at line : " << i << std::endl;
			assert(false);
		}
	}

	auto vocab_text = fileopen(base_pass + VOCAB_FILENAME);

	sig::for_each([&](uint i, std::wstring const& e){
		auto word = std::make_shared<std::wstring>(e);
		words_.push_back(word);
		word2id_.emplace(word, i);
	}
	, 0, vocab_text);

/*
	std::ifstream ifs(token_pass);
	std::string line;
	uint line_num = 0, token_ct = 0;

	if(!ifs){
		std::wcout << L"token file not found : " << token_pass << std::endl;
		return;
	}

	// Get feature size
	std::getline(ifs, line);
	doc_num_ = std::stoi(line);
	std::getline(ifs, line);
	uint wnum = std::stoi(line);
	words_.reserve(wnum);
	std::getline(ifs, line);
	int tnum = std::stoi(line);
	tokens_.reserve(tnum);

	std::cout << "document_num: " << doc_num_ << "\nword_num:" << wnum << "\ntoken_num:" << tnum << std::endl;
	
	if(doc_num_ <= 0 || tnum <= 0 || wnum <= 0) {
		std::cout << "token file is corrupted" << std::endl;
		return;
	}
	
	while(std::getline(ifs, line)){
		if( parseLine(line, token_ct) == - 1 && line != "") {
			std::cout << "error in token file at line : " << line_num << std::endl;
			return;
		}
	}


	auto vocab_pass = base_pass + VOCAB_FILENAME;
	std::ifstream ifs2(vocab_pass);
	for(uint i = 0; i < wnum; ++i) {
		std::getline(ifs2, line);
		auto wsp = std::make_shared<std::wstring>( sig::STRtoWSTR(line) );
		words_.push_back(wsp);
		word2id_.emplace(*wsp, i);
	}
*/
}

void InputData::save(FilepassString folder_pass)
{
	auto base_pass = sig::impl::modify_dirpass_tail(folder_pass, true);

	auto vocab_pass = base_pass + VOCAB_FILENAME;
	auto token_pass = base_pass + TOKEN_FILENAME;

	sig::clear_file(vocab_pass);
	sig::clear_file(token_pass);
	
	// save words
	std::wofstream ofs2(vocab_pass);
	for (auto const& word : words_){
		ofs2 << *word << std::endl;
	}
	ofs2.close();

	// save tokens
	std::ofstream ofs(token_pass);
	ofs << doc_num_ << std::endl;
	ofs << words_.size() << std::endl;
	ofs << tokens_.size() << std::endl;

	std::vector< std::unordered_map<uint, uint> > d_w_ct(doc_num_);

	for (auto const& token : tokens_){
		if (d_w_ct[token->doc_id].count(token->word_id)) ++d_w_ct[token->doc_id][token->word_id];
		else d_w_ct[token->doc_id].emplace(token->word_id, 1);
	}

	for (int d = 0; d < doc_num_; ++d){
		for (auto const& w2ct : d_w_ct[d]){
			ofs << (d + 1) << " " << (w2ct.first + 1) << " " << w2ct.second << std::endl;
		}
	}
}

}	//namespace sigtm