#include "input_text.h"

namespace sigtm
{

#if SIG_USE_NLP

void InputData::makeData(Documents const& raw_texts, std::wstring const& folder_pass)
{

	const auto ParallelFunc = [this](Document document){
		std::vector<std::wstring> result;
		auto& mecab = signlp::MecabWrapper::GetInstance();

		for (auto& sentence : document){
			_filter._pre_filter(sentence);		//形態素解析前フィルタ処理

			//形態素解析処理
			auto parsed = [&]{
				if (_filter._base_form) return mecab.ParseGenkeiAndFilter(sentence, [&](WordClass wc){ return _filter._selected_word_class.count(wc); });
				else return mecab.ParseSimpleAndFilter(sentence, [&](WordClass wc){ return _filter._selected_word_class.count(wc); });
			}();

			for (auto& word : parsed){
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

	for (auto const& document : raw_texts){
		results.push_back(std::async(std::launch::async, ParallelFunc, document));
	}

	Documents doc_words;

	for (auto& result : results){
		doc_words.push_back(result.get());
	}

	int token_ct = 0;
	int doc_id = -1;

	for (auto const& words : doc_words){
		++doc_id;
		for (auto& word : words){
			//wordが既出か判定
			if (word2id_.count(word)){
				tokens_.push_back(std::make_shared<Token>(token_ct, doc_id, word2id_[word]));
				++token_ct;
			}
			else{
				word2id_.emplace(word, words_.size());
				tokens_.push_back(std::make_shared<Token const>(token_ct, doc_id, word2id_[word]));
				words_.push_back(std::make_shared<std::wstring const>(word));
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
	std::ofstream ofs2(pass + VOCAB_FILENAME);
	for (auto const& word : words_){
		ofs2 << sig::WSTRtoSTR(*word) << std::endl;
	}

	//save token
	[&](){
		std::ofstream ofs(pass + TOKEN_FILENAME);
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
	}();
}
#endif

}
	