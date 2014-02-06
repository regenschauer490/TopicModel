#include <future>
#include <sstream>
#include "input_container.h"

namespace sigdm{
		
#if USE_MECAB

void InputDataFactory::MakeData_(Documents const& raw_texts, std::wstring const& folder_pass)
{

	const auto ParallelFunc = [this](Document document){
		std::vector<std::wstring> result;
		auto& mecab = signlp::MecabWrapper::GetInstance();

		for(auto& sentence : document){
			_filter._pre_filter(sentence);		//形態素解析前フィルタ処理
	
			//形態素解析処理
			auto parsed = [&]{
				if(_filter._base_form) return mecab.ParseGenkeiAndFilter(sentence, [&](WordClass wc){ return _filter._selected_word_class.count(wc); });
				else return mecab.ParseSimpleAndFilter(sentence, [&](WordClass wc){ return _filter._selected_word_class.count(wc); });
			}();

			for(auto& word : parsed){
				_filter._aft_filter(word);	//形態素解析後フィルタ処理
			}

			sig::RemoveAll(parsed, L"");

			std::move(parsed.begin(), parsed.end(), std::back_inserter(result));
		}

		return std::move(result);
	};

	std::cout << "make token data : " << std::endl;
	sig::TimeWatch tw;

	std::vector< std::future< std::vector<std::wstring> > > results;

	for(auto const& document : raw_texts){
		results.push_back( std::async(std::launch::async, ParallelFunc, document) );
	}

	Documents doc_words;

	for(auto& result : results){
		doc_words.push_back(result.get());
	}

	int token_ct = 0;
	int doc_id = -1;

	for(auto const& words : doc_words){ 
		++doc_id;
		for(auto& word : words){
			//wordが既出か判定
			if(_word2id_map.count(word)){	
				_tokens.push_back( std::make_shared<Token>(token_ct, doc_id, _word2id_map[word]) );
				++token_ct;
			}
			else{
				_word2id_map.emplace(word, _words.size());
				_tokens.push_back( std::make_shared<Token const>(token_ct, doc_id, _word2id_map[word]) );
				_words.push_back( std::make_shared<std::wstring const>(word) );
				++token_ct;
			}
		}
	}

/*			//指定語彙の除去
			if(!_filter->_excepted_words.empty()){
				_filter->_excepted_words[doc_id].count(

				std::wstring wqstr, tmp;
				const auto& et = _filter->_except_terms[doc_id];

				auto snit = et.begin();
				if(snit != et.end()){
					tmp = sig::STRtoWSTR(*snit);
					wqstr = L"(" + tmp;
					++snit;
					for(auto& snend = et.end(); snit != snend; ++snit){
						tmp = sig::STRtoWSTR(*snit);
						wqstr = (wqstr + L"|" + tmp);
					}
					wqstr = wqstr + L")";

					std::wregex qreg(wqstr);
					std::wstring rep = std::regex_replace(wline, qreg, std::wstring(L""));
					wline = rep;
				}
			}
*/

	auto pass = sig::DirpassTailModify(folder_pass, true);
	std::ofstream ofs2(pass +  VOCAB_FILENAME);
	for(auto const& word : _words){
		ofs2 << sig::WSTRtoSTR(*word) << std::endl;
	}
	
	//save token
	[&](){
		std::ofstream ofs(pass + TOKEN_FILENAME);
		ofs << _doc_num << std::endl;
		ofs << _words.size() << std::endl;
		ofs << _tokens.size() << std::endl;

		std::vector< std::unordered_map<uint, uint> > d_w_ct(_doc_num);

		for(auto const& token : _tokens){
			if(d_w_ct[token->doc_id].count(token->word_id)) ++d_w_ct[token->doc_id][token->word_id];
			else d_w_ct[token->doc_id].emplace(token->word_id, 1);
		}

		for(int d=0; d < _doc_num; ++d){
			for(auto const& w2ct : d_w_ct[d]){
				ofs << (d+1) << " " << (w2ct.first+1) << " " << w2ct.second << std::endl;
			}
		}
	}();
}

#endif

inline int InputDataFactory::ParseLine_(std::string const& line, uint& tct)
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
		_tokens.push_back( std::make_shared<Token>(tct, doc_id-1, word_id-1) );
		++tct;
	}

	return 0;
}

void InputDataFactory::ReconstructData_(std::wstring const& folder_pass)
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
	_doc_num = std::stoi(line);
	std::getline(ifs, line);
	uint wnum = std::stoi(line);
	_words.reserve(wnum);
	std::getline(ifs, line);
	int tnum = std::stoi(line);
	_tokens.reserve(tnum);

	std::cout << "document_num: " << _doc_num << "\nword_num:" << wnum << "\ntoken_num:" << tnum << std::endl;
	
	if(_doc_num <= 0 || tnum <= 0 || wnum <= 0) {
		std::cout << "token file is corrupted" << std::endl;
		return;
	}
	
	while(std::getline(ifs, line)){
		if( ParseLine_(line, token_ct) == - 1 && line != "") {
			std::cout << "error in token file at line : " << line_num << std::endl;
			return;
		}
	}


	auto vocab_pass = base_pass + VOCAB_FILENAME;
	std::ifstream ifs2(vocab_pass);
	for(uint i = 0; i < wnum; ++i) {
		std::getline(ifs2, line);
		auto wsp = std::make_shared<std::wstring>( sig::STRtoWSTR(line) );
		_words.push_back(wsp);
		_word2id_map.emplace(*wsp, i);
	}
}

}	//namespace sigdm