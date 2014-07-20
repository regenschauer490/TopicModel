/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#include "process_word.h"

namespace signlp{

using sig::wstr_to_str;
using sig::str_to_wstr;
using sig::split;

EvaluationLibrary::EvaluationLibrary() : mecab_(MecabWrapper::getInstance())
{
	auto PNE = [](std::string const& s)->PosiNega{
		if (s == "p") return PosiNega::P;
		if (s == "n") return PosiNega::N;
		if (s == "e") return PosiNega::E;
		return PosiNega::NA;
	};

	auto PNS = [](std::string const& s)->PNStandard{
		if (s == "評価・感情/主観") return PNStandard::EvaEmo_Sbj;
		if (s == "評価") return PNStandard::EvaEmo_Sbj;
		if (s == "状態/客観") return PNStandard::State_Obj;
		if (s == "存在・性質") return PNStandard::ExisProp;
		if (s == "経験") return PNStandard::Exp;
		if (s == "出来事") return PNStandard::Event;
		if (s == "行為") return PNStandard::Act;
		if (s == "場所") return PNStandard::Place;
		return PNStandard::NA;
	};

	std::string line;
	std::wstring wstr;
		 
	auto read1 = sig::read_line<std::string>(evaluation_word_filepass);

	if (! sig::is_container_valid(read1)){
		std::wcerr << L"Cannot open : " << evaluation_word_filepass << std::endl;
	}
	else{
		for (auto const& line : sig::fromJust(read1)){
			const auto linevec = split(line, ",");

			double score;
			try{
				score = stod(linevec[3]);
			}
			catch (std::exception& e){
				std::cout << "exception in evlib ctor at stod() : " << typeid(e).name() << std::endl;
				score = 0.0;
			}

			if (abs(score) < evaluation_word_threshold) continue;

			word_ev_[linevec[0]] = std::make_tuple(score, StrToWC(linevec[2]));
		}
	}

	auto read2 = sig::read_line<std::string>(evaluation_noun_filepass);

	if (! sig::is_container_valid(read2)){
		std::wcerr << L"Cannot open : " << evaluation_noun_filepass << std::endl;
	}
	else{
		for (auto const& line : sig::fromJust(read2)){
			const auto linevec = split(line, ",");

			noun_ev_[linevec[0]] = std::make_tuple(PNE(linevec[1]), PNS(linevec[3]));
		}
	}

	auto read3 = sig::read_line<std::string>(evaluation_declinable_filepass);

	if (! sig::is_container_valid(read3)){
		std::wcerr << L"Cannot open : " << evaluation_declinable_filepass << std::endl;
	}
	else{
		for (auto const& line : sig::fromJust(read3)){
			const auto linevec = split(line, ",");

			wstr = str_to_wstr(linevec[2]);
			std::wregex reg(L"\\s");
			wstr = std::regex_replace(wstr, reg, L"");
			declinable_ev_[wstr_to_str(wstr)] = std::make_tuple(PNE(linevec[0]), PNS(linevec[1]));
		}
	}
}

PosiNega EvaluationLibrary::getSentencePN(std::string const& sentence, uint threshold) const
{
	double score = 0;
	const auto pws = mecab_.parseGenkei(sentence);

	for (auto& pw : pws){
		for (auto& ev : word_ev_){
			if (pw == ev.first){
				score += std::get<0>(ev.second);
			}
		}
	}

	//std::cout << score << std::endl;

	if (score > threshold) return PosiNega::P;
	else if (score < (-1)*threshold) return PosiNega::N;
	else return PosiNega::E;
}
PosiNega EvaluationLibrary::getSentencePN(std::wstring const& sentence, uint threshold) const
{
	return getSentencePN(wstr_to_str(sentence), threshold);
}

PosiNega EvaluationLibrary::getWordPN(std::string const& word, WordClass wc) const
{
	switch (wc){
	case WordClass::名詞:
	{
		const auto ser = noun_ev_.find(word);
		if (ser != noun_ev_.end()) return std::get<0>(ser->second);
		else return PosiNega::NA;
	}
	case WordClass::動詞:
	case WordClass::形容詞:
	{
		const auto ser = declinable_ev_.find(word);
		if (ser != declinable_ev_.end()) return std::get<0>(ser->second);
		else return PosiNega::NA;
	}
	default:
		return PosiNega::NA;
	}
}
PosiNega EvaluationLibrary::getWordPN(std::wstring const& word, WordClass wc) const
{
	return getWordPN(wstr_to_str(word), wc);
}

PosiNega EvaluationLibrary::getSentencePN(std::string const& sentence, ScoreMap score_map, double th) const
{
	if (!score_map.count(WordClass::名詞)) score_map[WordClass::名詞] = 0;
	if (!score_map.count(WordClass::形容詞)) score_map[WordClass::形容詞] = 0;
	if (!score_map.count(WordClass::動詞)) score_map[WordClass::動詞] = 0;
	if (!score_map.count(WordClass::NA)) score_map[WordClass::NA] = 0;

	int score = 0;

	auto Noun_func = [&](const std::vector<std::tuple<std::string, WordClass>>& parsed_sentence)->int{
		int tscore = 0;

		for (const auto& w : parsed_sentence){
			for (const auto& ev : noun_ev_){
				if (std::get<0>(w) == ev.first){
					const auto pn = std::get<0>(ev.second);
					if (pn == PosiNega::P)  tscore += score_map[WordClass::名詞];
					else if (pn == PosiNega::N) tscore -= score_map[WordClass::名詞];
				}
			}
		}
		return tscore;
	};

	auto Declinable_func = [&](const std::vector<std::tuple<std::string, WordClass>>& parsed_sentence){
		int tscore = 0;

		for (const auto& w : parsed_sentence){
			for (auto& ev : declinable_ev_){
				if (std::get<0>(w) == ev.first){
					const auto pn = std::get<0>(ev.second);
					if (pn == PosiNega::P) tscore += score_map[std::get<1>(w)];
					else if (pn == PosiNega::N) tscore -= score_map[std::get<1>(w)];
				}
			}
		}
		/*
		wsmatch m;

		for(auto& decli : declinable_ev_){
		const std::wregex reg(decli.first);
		if( regex_search(sentence, m, reg) ){
		const auto pn = std::get<0>(decli.second);
		const auto pws = MecabWrapper::parseGenkeiWithWC( wstr_to_str(decli.first) );
		if(pws.empty()) continue;

		if( pn == PosiNega::P ) tscore += score_map[std::get<1>(pws[0])];
		else if( pn == PosiNega::N ) tscore -= score_map[std::get<1>(pws[0])];
		}
		}
		*/
		return tscore;
	};

	const auto parse = mecab_.parseGenkeiWithWC(sentence);
	std::vector<std::tuple<std::string, WordClass>> parse_n, parse_d;
	std::copy_if(parse.begin(), parse.end(), back_inserter(parse_n), [](const std::tuple<std::string, WordClass>& e){ return std::get<1>(e) == WordClass::名詞; });
	std::copy_if(parse.begin(), parse.end(), back_inserter(parse_d), [](const std::tuple<std::string, WordClass>& e){ return std::get<1>(e) == WordClass::形容詞 || std::get<1>(e) == WordClass::動詞; });

	//auto nf = async(launch::async, );
	//auto df = async(launch::async, );

	score += Noun_func(parse_n) + Declinable_func(parse_d);

	//std::cout << score << std::endl;

	th = abs(th);
	if (score > th) return PosiNega::P;
	if (score < -th) return PosiNega::N;
	return PosiNega::E;
}
PosiNega EvaluationLibrary::getSentencePN(std::wstring const& sentence, ScoreMap score_map, double th) const
{
	return getSentencePN(wstr_to_str(sentence), score_map, th);
}

PNStandard EvaluationLibrary::getPNStandard(std::string const& str) const
{
	const auto ser = noun_ev_.find(str);
	if (ser != noun_ev_.end()) return std::get<1>(ser->second);
	return PNStandard::NA;
}
PNStandard EvaluationLibrary::getPNStandard(std::wstring const& word) const
{
	return getPNStandard(wstr_to_str(word));
}

}