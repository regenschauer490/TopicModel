#ifndef SIG_POLAR_EVALUATION_HPP
#define SIG_POLAR_EVALUATION_HPP

#include <unordered_map>
#include "mecab_wrapper.hpp"
#include "signlp.hpp"


namespace signlp
{
static const wchar_t* evaluation_word_filepass = L"J_Evaluation_lib/単語感情極性対応表.csv"; //evword.csv";
static const auto evaluation_noun_filepass = std::wstring(L"J_Evaluation_lib/noun.csv");
static const auto evaluation_declinable_filepass = std::wstring(L"J_Evaluation_lib/declinable.csv");
static const double evaluation_word_threshold = 0.80;		//日本語評価極性辞書で使用する語彙の閾値設定(|スコア|>閾値)


typedef std::unordered_map<WordClass, unsigned> ScoreMap;

/* 日本語評価極性辞書を扱うシングルトンクラス */
class EvaluationLibrary{
	//単語感情極性対応表 [-1～1のスコア, 品詞]
	std::unordered_map< std::string, std::tuple<double, WordClass> > word_ev_;
	//名詞,用言のP/N判定 [P/N, 評価基準]
	std::unordered_map< std::string, std::tuple<PosiNega, PNStandard> > noun_ev_;
	std::unordered_map< std::string, std::tuple<PosiNega, PNStandard> > declinable_ev_;

	MecabWrapper& mecab_;

private:
	EvaluationLibrary();

	EvaluationLibrary(const EvaluationLibrary&) = delete;
public:
	static EvaluationLibrary& getInstance(){
		static EvaluationLibrary instance;
		return instance;
	}

	/* 単語感情極性対応表使用 */
	//文章のP/Nを判定 (各単語-1～1のスコアが付与されており,合計を求める. N: total < -threshold，P: threshold < total) 
	PosiNega getSentencePN(std::string const& sentence, uint threshold = 0.8) const;
	PosiNega getSentencePN(std::wstring const& sentence, uint threshold = 0.8) const;


	/* 名詞,用言のP/N判定使用 */
	//単語のP/Nを取得
	PosiNega getWordPN(std::string const&  word, WordClass wc) const;
	PosiNega getWordPN(std::wstring const& word, WordClass wc) const;
	
	//文章のP/Nを判定 (品詞別のスコアと閾値を任意に与える) 
	PosiNega getSentencePN(std::string const&  sentence, ScoreMap score_map, double th) const;
	PosiNega getSentencePN(std::wstring const& sentence, ScoreMap score_map, double th) const;
		
	//その単語のP/Nの評価基準を取得
	PNStandard getPNStandard(std::string const&  word) const;
	PNStandard getPNStandard(std::wstring const& word) const;
};


/* implementation */

inline EvaluationLibrary::EvaluationLibrary() : mecab_(MecabWrapper::getInstance())
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


	auto read1 = sig::load_line(evaluation_word_filepass);

	if (!sig::isJust(read1)){
		std::wcerr << L"Cannot open : " << evaluation_word_filepass << std::endl;
	}
	else{
		for (auto const& line : sig::fromJust(read1)){
			const auto linevec = sig::split(line, ",");

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

	auto read2 = sig::load_line(evaluation_noun_filepass);

	if (!sig::isJust(read2)){
		std::wcerr << L"Cannot open : " << evaluation_noun_filepass << std::endl;
	}
	else{
		for (auto const& line : sig::fromJust(read2)){
			const auto linevec = sig::split(line, ",");

			noun_ev_[linevec[0]] = std::make_tuple(PNE(linevec[1]), PNS(linevec[3]));
		}
	}

	auto read3 = sig::load_line(evaluation_declinable_filepass);

	if (!sig::isJust(read3)){
		std::wcerr << L"Cannot open : " << evaluation_declinable_filepass << std::endl;
	}
	else{
		for (auto const& line : sig::fromJust(read3)){
			const auto linevec = sig::split(line, ",");

			auto wstr = sig::str_to_wstr(linevec[2]);
			std::wregex reg(L"\\s");
			wstr = std::regex_replace(wstr, reg, L"");
			declinable_ev_[sig::wstr_to_str(wstr)] = std::make_tuple(PNE(linevec[0]), PNS(linevec[1]));
		}
	}
}

inline PosiNega EvaluationLibrary::getSentencePN(std::string const& sentence, uint threshold) const
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
inline PosiNega EvaluationLibrary::getSentencePN(std::wstring const& sentence, uint threshold) const
{
	return getSentencePN(sig::wstr_to_str(sentence), threshold);
}

inline PosiNega EvaluationLibrary::getWordPN(std::string const& word, WordClass wc) const
{
	switch (wc){
	case WordClass::Noun:
	{
						  const auto ser = noun_ev_.find(word);
						  if (ser != noun_ev_.end()) return std::get<0>(ser->second);
						  else return PosiNega::NA;
	}
	case WordClass::Verb:
	case WordClass::Adjective:
	{
						   const auto ser = declinable_ev_.find(word);
						   if (ser != declinable_ev_.end()) return std::get<0>(ser->second);
						   else return PosiNega::NA;
	}
	default:
		return PosiNega::NA;
	}
}
inline PosiNega EvaluationLibrary::getWordPN(std::wstring const& word, WordClass wc) const
{
	return getWordPN(sig::wstr_to_str(word), wc);
}

inline PosiNega EvaluationLibrary::getSentencePN(std::string const& sentence, ScoreMap score_map, double th) const
{
	if (!score_map.count(WordClass::Noun)) score_map[WordClass::Noun] = 0;
	if (!score_map.count(WordClass::Adjective)) score_map[WordClass::Adjective] = 0;
	if (!score_map.count(WordClass::Verb)) score_map[WordClass::Verb] = 0;
	if (!score_map.count(WordClass::NA)) score_map[WordClass::NA] = 0;

	int score = 0;

	auto Noun_func = [&](const std::vector<std::tuple<std::string, WordClass>>& parsed_sentence)->int{
		int tscore = 0;

		for (const auto& w : parsed_sentence){
			for (const auto& ev : noun_ev_){
				if (std::get<0>(w) == ev.first){
					const auto pn = std::get<0>(ev.second);
					if (pn == PosiNega::P)  tscore += score_map[WordClass::Noun];
					else if (pn == PosiNega::N) tscore -= score_map[WordClass::Noun];
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
		const auto pws = MecabWrapper::parseGenkeiWithWC( sig::wstr_to_str(decli.first) );
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
	std::copy_if(parse.begin(), parse.end(), back_inserter(parse_n), [](const std::tuple<std::string, WordClass>& e){ return std::get<1>(e) == WordClass::Noun; });
	std::copy_if(parse.begin(), parse.end(), back_inserter(parse_d), [](const std::tuple<std::string, WordClass>& e){ return std::get<1>(e) == WordClass::Adjective || std::get<1>(e) == WordClass::Verb; });

	//auto nf = async(launch::async, );
	//auto df = async(launch::async, );

	score += Noun_func(parse_n) + Declinable_func(parse_d);

	//std::cout << score << std::endl;

	th = abs(th);
	if (score > th) return PosiNega::P;
	if (score < -th) return PosiNega::N;
	return PosiNega::E;
}
inline PosiNega EvaluationLibrary::getSentencePN(std::wstring const& sentence, ScoreMap score_map, double th) const
{
	return getSentencePN(sig::wstr_to_str(sentence), score_map, th);
}

inline PNStandard EvaluationLibrary::getPNStandard(std::string const& str) const
{
	const auto ser = noun_ev_.find(str);
	if (ser != noun_ev_.end()) return std::get<1>(ser->second);
	return PNStandard::NA;
}
inline PNStandard EvaluationLibrary::getPNStandard(std::wstring const& word) const
{
	return getPNStandard(sig::wstr_to_str(word));
}

}
#endif