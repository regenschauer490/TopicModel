/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#include "input_text.h"
#include "SigUtil/lib/modify.hpp"
#include <future>

namespace sigtm
{

#if USE_SIGNLP

void InputDataFromText::makeData(Documents const& raw_texts)
{

	const auto ParallelFunc = [](Document document, FilterSetting const& filter){
		std::vector<std::wstring> result;
		auto& mecab = signlp::MecabWrapper::getInstance();

		for (auto& sentence : document){
			filter.pre_filter_(sentence);		//�`�ԑf��͑O�t�B���^����

			//�`�ԑf��͏���
			auto parsed = [&]{
				if (filter.base_form_) return mecab.parseGenkeiThroughFilter(sentence, [&](WordClass wc){ return filter.selected_word_class_.count(wc); });
				else return mecab.parseSimpleThroughFilter(sentence, [&](WordClass wc){ return filter.selected_word_class_.count(wc); });
			}();

			for (auto& word : parsed){
				filter.aft_filter_(word);	//�`�ԑf��͌�t�B���^����
			}

			sig::remove_all(parsed, L"");

			std::move(parsed.begin(), parsed.end(), std::back_inserter(result));
		}

		return std::move(result);
	};

	std::cout << "make token data : " << std::endl;
	sig::TimeWatch tw;

	std::vector< std::future< std::vector<std::wstring> > > results;

	for (auto const& document : raw_texts){
		results.push_back(std::async(std::launch::async, ParallelFunc, document, filter_));
	}

	Documents doc_words;

	for (auto& result : results){
		doc_words.push_back(result.get());
	}
	for (uint i = 0; i<doc_words.size(); ++i) std::wcout << doc_names_[i] << L" parsed. word-num: " << doc_words[i].size() << std::endl;
	std::cout << std::endl;

	int token_ct = 0;
	int doc_id = -1;

	for (auto const& words : doc_words){
		++doc_id;
		for (auto& word : words){
			auto wp = std::make_shared<std::wstring const>(word);

			//word�����o������
			if (words_.hasElement(wp)){
				tokens_.push_back(Token(token_ct, doc_id, words_.getWordID(wp)));
				++token_ct;
			}
			else{
				uint index = words_.size();
				words_.emplace(index, wp);
				tokens_.push_back(Token(token_ct, doc_id, index));
				++token_ct;
			}
		}
	}

	/*			//�w���b�̏���
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
	