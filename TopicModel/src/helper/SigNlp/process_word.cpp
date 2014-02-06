#include "process_word.h"

namespace signlp{
	using sig::WSTRtoSTR;
	using sig::STRtoWSTR;

	EvaluationLibrary::EvaluationLibrary() : _mecabw(MecabWrapper::GetInstance())
	{
		auto PNE = [](std::string const& s)->PosiNega{
			if(s == "p") return PosiNega::P; 
			if(s == "n") return PosiNega::N;
			if(s == "e") return PosiNega::E;
			return PosiNega::_NA;
		};

		auto PNS = [](std::string const& s)->PNStandard{
			if(s == "評価・感情/主観") return PNStandard::EvaEmo_Sbj;
			if(s == "評価") return PNStandard::EvaEmo_Sbj;
			if(s == "状態/客観") return PNStandard::State_Obj;
			if(s == "存在・性質") return PNStandard::ExisProp;
			if(s == "経験") return PNStandard::Exp;
			if(s == "出来事") return PNStandard::Event;
			if(s == "行為") return PNStandard::Act;
			if(s == "場所") return PNStandard::Place;
			return PNStandard::_NA;
		};

		std::string line;
		std::wstring wstr;

		std::ifstream ifs0(evaluation_word_filepass);
		if(!ifs0){
			std::cerr << "Cannot open : " << evaluation_word_filepass << std::endl;
			goto NEXT1;
		}
		int ct = 0;
		while(!ifs0.eof()){
			getline(ifs0, line);
			if(line.empty()) continue;

			const auto linevec = sig::Split(line, ",");
			double sc;
			try{
				sc = stod(linevec[3]);
			}
			catch(std::exception& e){
				std::cout << "exception in evlib ctor at stod() : " << typeid(e).name() << std::endl;
				sc = 0.0;
			}

			if(abs(sc) < evaluation_word_threshold) continue;

			_word_ev[linevec[0]] = std::make_tuple( sc, StrToWC(linevec[2]) );
		}

	NEXT1:
		std::ifstream ifs1(evaluation_noun_filepass);
		if(!ifs1){
			std::cerr << "Cannot open : " << evaluation_noun_filepass << std::endl;
			goto NEXT2;
		}
		while(!ifs1.eof()){
			getline(ifs1, line);
			if(line.empty()) continue;

			const auto linevec = sig::Split(line, ",");
			_noun_ev[linevec[0]] = std::make_tuple( PNE(linevec[1]),  PNS(linevec[3]) );
		}

	NEXT2:
		std::ifstream ifs2(evaluation_declinable_filepass);
		if(!ifs2){
			std::cerr << "Cannot open : " << evaluation_declinable_filepass << std::endl;
			return;
		}
		while(!ifs2.eof()){
			getline(ifs2, line);
			if(line.empty()) continue;

			const auto linevec = sig::Split(line, ",");
			wstr = STRtoWSTR(linevec[2]);
			std::wregex reg(L"\\s");
			wstr = std::regex_replace(wstr, reg, L"");
			_declinable_ev[WSTRtoSTR(wstr)] = std::make_tuple( PNE(linevec[0]), PNS(linevec[1]) );
		}
	}

	PosiNega EvaluationLibrary::GetSentencePosiNega(std::string const& sentence) const
	{
		const int threshold = 0.8;	//合計スコアのP/N判定閾値

		double score = 0;
		const auto pws = _mecabw.ParseGenkei(sentence);
		
		for(auto& pw : pws){
			for(auto& ev : _word_ev){
				if( pw == ev.first ){
					score += std::get<0>(ev.second);
				}
			}
		}

		//std::cout << score << std::endl;

		if(score > threshold) return PosiNega::P;
		if(score < (-1)*threshold) return PosiNega::N;
		return PosiNega::E;
	}
	PosiNega EvaluationLibrary::GetSentencePosiNega(std::wstring const& sentence) const{
		auto str = WSTRtoSTR(sentence); return GetSentencePosiNega(str);
	}

	PosiNega EvaluationLibrary::GetWordPosiNega(std::string const& word, WordClass wc) const
	{
		switch(wc){
			case WordClass::名詞 : {
				const auto ser = _noun_ev.find(word);
				if(ser != _noun_ev.end()) return std::get<0>(ser->second);
				else return PosiNega::_NA;
								   }
			case WordClass::動詞 :
				break;
			case WordClass::形容詞 :

			default :
				return PosiNega::_NA;
		}
	}
	PosiNega EvaluationLibrary::GetWordPosiNega(std::wstring const& word, WordClass wc) const{
		auto str = WSTRtoSTR(word);
		return GetWordPosiNega(str, wc); 
	}

	PosiNega EvaluationLibrary::GetSentencePosiNega(std::string const& sentence, ScoreMap score_map, double th) const
	{
		if(!score_map.count(WordClass::名詞)) score_map[WordClass::名詞] = 0;
		if(!score_map.count(WordClass::形容詞)) score_map[WordClass::形容詞] = 0;
		if(!score_map.count(WordClass::動詞)) score_map[WordClass::動詞] = 0;
		if(!score_map.count(WordClass::_NA)) score_map[WordClass::_NA] = 0;

		int score = 0;

		auto Noun_func = [&](const std::vector<std::tuple<std::string,WordClass>>& parsed_sentence)->int{
			int tscore = 0;

			for(const auto& w : parsed_sentence){
				for(const auto& ev : _noun_ev){
					if( std::get<0>(w) == ev.first ){
						const auto pn = std::get<0>(ev.second);
						if( pn == PosiNega::P )  tscore += score_map[WordClass::名詞];
						else if( pn == PosiNega::N ) tscore -= score_map[WordClass::名詞];
					}
				}
			}
			return tscore;
		};

		auto Declinable_func = [&](const std::vector<std::tuple<std::string,WordClass>>& parsed_sentence){
			int tscore = 0;

			for(const auto& w : parsed_sentence){
				for(auto& ev : _declinable_ev){
					if( std::get<0>(w) == ev.first ){
						const auto pn = std::get<0>(ev.second);
						if( pn == PosiNega::P ) tscore += score_map[std::get<1>(w)];
						else if( pn == PosiNega::N ) tscore -= score_map[std::get<1>(w)];
					}
				}
			}
			/*
			wsmatch m;

			for(auto& decli : _declinable_ev){
				const std::wregex reg(decli.first);
				if( regex_search(sentence, m, reg) ){
					const auto pn = std::get<0>(decli.second);
					const auto pws = MecabWrapper::ParseGenkeiWithWC( WSTRtoSTR(decli.first) );
					if(pws.empty()) continue;

					if( pn == PosiNega::P ) tscore += score_map[std::get<1>(pws[0])];
					else if( pn == PosiNega::N ) tscore -= score_map[std::get<1>(pws[0])];
				}
			}
			*/
			return tscore;
		};

		const auto parse = _mecabw.ParseGenkeiWithWC(sentence);
		std::vector<std::tuple<std::string,WordClass>> parse_n, parse_d;
		std::copy_if(parse.begin(), parse.end(), back_inserter(parse_n), [](const std::tuple<std::string,WordClass>& e){ return std::get<1>(e) == WordClass::名詞; });
		std::copy_if(parse.begin(), parse.end(), back_inserter(parse_d), [](const std::tuple<std::string,WordClass>& e){ return std::get<1>(e) == WordClass::形容詞 || std::get<1>(e) == WordClass::動詞; });

		//auto nf = async(launch::async, );
		//auto df = async(launch::async, );

		score += Noun_func(parse_n) + Declinable_func(parse_d);

		//std::cout << score << std::endl;

		th = abs(th);
		if(score > th) return PosiNega::P;
		if(score < -th) return PosiNega::N;
		return PosiNega::E;
	}
	PosiNega EvaluationLibrary::GetSentencePosiNega(std::wstring const& sentence, ScoreMap score_map, double th) const{
		auto str = WSTRtoSTR(sentence);
		return GetSentencePosiNega(str, score_map, th); 
	}

	PNStandard EvaluationLibrary::GetPNStandard(std::string const& str) const
	{
		const auto ser = _noun_ev.find(str);
		if(ser != _noun_ev.end()) return std::get<1>(ser->second);
		return PNStandard::_NA;
	}
	PNStandard EvaluationLibrary::GetPNStandard(std::wstring const& word) const{
		auto str = WSTRtoSTR(word); 
		return GetPNStandard(str); 
	}

}