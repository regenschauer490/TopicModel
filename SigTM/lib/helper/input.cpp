#include <future>
#include <sstream>
#include "input.h"

namespace sigtm
{
	
inline int InputData::parseLine(std::string const& line, uint& tct)
{
	std::istringstream is(line);
	int doc_id = 0;
	int word_id = 0;
	int count = 0;

	is >> doc_id >> word_id >> count;
	if (!doc_id || !word_id || !count) {
		std::cout << "parse error";
		return -1;
	}
  
	for(int i = 0; i < count; ++i){
		tokens_.push_back( std::make_shared<Token>(tct, doc_id-1, word_id-1) );
		++tct;
	}

	return 0;
}

void InputData::reconstruct(std::wstring const& folder_pass)
{
	auto filenames = sig::GetFileNames(folder_pass, false);
	if(!filenames){
		std::wcout << L"folder not found : " << folder_pass << std::endl;
		return;
	}

	auto base_pass = sig::DirpassTailModify(folder_pass, true);
	auto token_pass = base_pass + TOKEN_FILENAME;
	std::ifstream ifs(token_pass);
	std::string line;
	uint line_num = 0, token_ct = 0;

	if(!ifs){
		std::cout << "token file not found : " << sig::WSTRtoSTR(token_pass) << std::endl;
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
		_word2id_map.emplace(*wsp, i);
	}
}

}	//namespace sigtm