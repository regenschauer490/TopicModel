/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#include "input_text.h"
#include "SigUtil/lib/modify/remove.hpp"
#include "SigUtil/lib/calculation/basic_statistics.hpp"
#include "SigUtil/lib/tools/time_watch.hpp"
#include <future>

namespace sigtm
{

#if USE_SIGNLP
void InputDataFromText::makeData(DocumentType type, Documents const& raw_texts)
{

	const auto ParallelFunc = [](Document document, FilterSetting const& filter){
		std::vector<std::vector<std::wstring>> result;
		auto& mecab = signlp::MecabWrapper::getInstance();

		for (auto& sentence : document){
			filter.pre_filter_(sentence);		//形態素解析前フィルタ処理

			//形態素解析処理
			auto parsed = [&]{
				if (filter.base_form_) return mecab.parseGenkeiThroughFilter(sentence, [&](WordClass wc){ return filter.selected_word_class_.count(wc); });
				else return mecab.parseSimpleThroughFilter(sentence, [&](WordClass wc){ return filter.selected_word_class_.count(wc); });
			}();

			for (auto& word : parsed){
				filter.aft_filter_(word);	//形態素解析後フィルタ処理
			}

			sig::remove_all(parsed, L"");

			result.push_back(std::vector<std::wstring>());
			std::move(parsed.begin(), parsed.end(), std::back_inserter(result.back()));
		}

		return std::move(result);
	};

	std::cout << "make token data : " << std::endl;
	sig::TimeWatch tw;

	std::vector< std::future< std::vector<std::vector<std::wstring>> > > results;

	for (auto const& document : raw_texts){
		results.push_back(std::async(std::launch::async, ParallelFunc, document, filter_));
	}

	std::vector<std::vector<std::vector<std::wstring>>> doc_line_words;

	for (auto& result : results){
		doc_line_words.push_back(result.get());
	}
	for (uint i = 0; i<doc_line_words.size(); ++i) std::wcout << doc_names_[i] << L" parsed. word-num: " << sig::sum(doc_line_words[i], [](std::vector<std::wstring> const& e){ return e.size(); }) << std::endl;
	std::cout << std::endl;

	int token_ct = 0;
	int doc_id = 0;
	for (auto const& line_words : doc_line_words){
		int line_id = 0;
		for (auto const& words : line_words){
			for (auto const& word : words){
				auto wp = std::make_shared<std::wstring const>(word);

				//wordが既出か判定
				if (words_.hasElement(wp)){
					if (DocumentType::Tweet == type) tokens_.push_back(Token(token_ct, doc_id, line_id, words_.getWordID(wp)));
					else tokens_.push_back(Token(token_ct, doc_id, words_.getWordID(wp)));
					++token_ct;
				}
				else{
					uint index = words_.size();
					words_.emplace(index, wp);
					if (DocumentType::Tweet == type) tokens_.push_back(Token(token_ct, doc_id, line_id, index));
					else tokens_.push_back(Token(token_ct, doc_id, index));
					++token_ct;
				}
			}
			++line_id;
		}
		++doc_id;
	}
	is_token_sorted_ = true;

	/*			//指定語彙の除去
	if(!filter_->excepted_words_.empty()){
	filter_->excepted_words_[doc_id].count(

	std::wstring wqstr, tmp;
	const auto& et = filter_->_except_terms[doc_id];

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
}
#endif

}
	