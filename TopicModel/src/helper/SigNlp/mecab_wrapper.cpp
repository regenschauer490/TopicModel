#include "mecab_wrapper.h"

namespace signlp{
	using sig::WSTRtoSTR;
	using sig::STRtoWSTR;

#define PARSE_WSTRING_VER(PARSE_ORIGINAL_FUNCTION) \
	std::vector< std::tuple<std::wstring, WordClass> > result; \
	\
	auto tmp = PARSE_ORIGINAL_FUNCTION(WSTRtoSTR(sentence)); \
	std::transform(tmp.begin(), tmp.end(), std::back_inserter(result), [](std::tuple<std::string, WordClass> const& e){ \
		return std::make_tuple(STRtoWSTR(std::get<0>(e)), std::get<1>(e)); \
	}); \
	\
	return std::move(result);


	volatile void MecabWrapper::ParseImpl(std::string const& src, std::string& dest) const
	{
		TaggerPtr tagger(_model->createTagger());
		LatticePtr lattice(_model->createLattice());
		/*	if(!tagger || !lattice){
		getchar();
		}*/
		lattice->set_sentence(src.c_str());

		tagger->parse(lattice.get());

		dest.assign(lattice->toString());
	}

	std::vector<std::string> MecabWrapper::ParseSimple(std::string const& sentence) const
	{
		std::vector<std::string> result;
		if (enable_warning && sentence.empty()){ std::cout << "sentense is empty" << std::endl; return result; }

		std::string parse;
		ParseImpl(sentence, parse);
		auto wlist = sig::Split(parse, "\n");
		for (auto& w : wlist){
			auto tmp = sig::Split(w, "\t");
			result.push_back(tmp[0]);
		}

		return std::move(result);
	}
	std::vector<std::wstring> MecabWrapper::ParseSimple(std::wstring const& sentence) const{
		return STRtoWSTR(ParseSimple(WSTRtoSTR(sentence)));
	}

	std::vector< std::tuple<std::string, WordClass> > MecabWrapper::ParseSimpleWithWC(std::string const& sentence) const
	{
		std::vector< std::tuple<std::string, WordClass> > result;
		if (enable_warning && sentence.empty()){ std::cout << "sentense is empty" << std::endl; return result; }

		std::string parse;
		ParseImpl(sentence, parse);
		auto wlist = sig::Split(parse, "\n");
		for (auto& w : wlist){
			auto tmp = sig::Split(w, "\t");
			if (tmp.size() < 2) continue;
			auto tmp2 = sig::Split(tmp[1], ",");
			if (tmp2.size() < 7) continue;
			result.push_back(std::make_tuple(tmp[0], StrToWC(tmp2[0])));
		}

		return std::move(result);
	}
	std::vector< std::tuple<std::wstring, WordClass> > MecabWrapper::ParseSimpleWithWC(std::wstring const& sentence) const
	{
		PARSE_WSTRING_VER(ParseSimpleWithWC);
	}

	std::vector<std::string> MecabWrapper::ParseGenkei(std::string const& sentence, bool skip) const
	{
		std::vector<std::string> result;
		if (enable_warning && sentence.empty()){ std::cout << "sentense is empty" << std::endl; return result; }

		std::string parse;
		ParseImpl(sentence, parse);
		auto wlist = sig::Split(parse, "\n");

		for (auto& w : wlist){
			auto tmp = sig::Split(w, "\t");
			if (tmp.size() < 2) continue;
			auto tmp2 = sig::Split(tmp[1], ",");
			if (tmp2.size() < 7) continue;
			if (tmp2[6] == "*"){
				if (!skip) result.push_back(tmp[0]);
			}
			else result.push_back(tmp2[6]);
		}

		return std::move(result);
	}
	std::vector<std::wstring> MecabWrapper::ParseGenkei(std::wstring const& sentence, bool skip) const{
		return STRtoWSTR(ParseGenkei(WSTRtoSTR(sentence)));
	}

	std::vector< std::tuple<std::string, WordClass> > MecabWrapper::ParseGenkeiWithWC(std::string const& sentence, bool skip) const
	{
		std::vector< std::tuple<std::string, WordClass> > result;
		if (enable_warning && sentence.empty()){ std::cout << "sentense is empty" << std::endl; return result; }

		std::string parse;
		ParseImpl(sentence, parse);

		auto wlist = sig::Split(parse, "\n");
		for (auto& w : wlist){
			auto tmp = sig::Split(w, "\t");
			if (tmp.size() < 2) continue;
			auto tmp2 = sig::Split(tmp[1], ",");
			if (tmp2.size() < 7) continue;
			if (tmp2[6] == "*"){
				if (!skip) result.push_back(std::make_tuple(tmp[0], StrToWC(tmp2[0])));
			}
			else result.push_back(std::make_tuple(tmp2[6], StrToWC(tmp2[0])));
		}

		return std::move(result);
	}
	std::vector< std::tuple<std::wstring, WordClass> > MecabWrapper::ParseGenkeiWithWC(std::wstring const& sentence, bool skip) const
	{
		PARSE_WSTRING_VER(ParseGenkeiWithWC);
	}

	std::vector<std::string> MecabWrapper::ParseSimpleAndFilter(std::string const& sentence, std::function<bool(WordClass)> const& pred) const
	{
		auto parse = ParseSimpleWithWC(sentence);

		std::vector<std::string> result;
		for (const auto& w : parse){
			if (pred(std::get<1>(w))) result.push_back(std::get<0>(w));
		}

		return std::move(result);
	}
	std::vector<std::wstring> MecabWrapper::ParseSimpleAndFilter(std::wstring const& sentence, std::function<bool(WordClass)> const& pred) const{
		return STRtoWSTR(ParseSimpleAndFilter(WSTRtoSTR(sentence), pred)); 
	}

	std::vector<std::string> MecabWrapper::ParseGenkeiAndFilter(std::string const& sentence, std::function<bool(WordClass)> const& pred) const
	{
		auto parse = ParseGenkeiWithWC(sentence);

		std::vector<std::string> result;
		for (const auto& w : parse){
			if (pred(std::get<1>(w))) result.push_back(std::get<0>(w));
		}

		return std::move(result);
	}
	std::vector<std::wstring> MecabWrapper::ParseGenkeiAndFilter(std::wstring const& sentence, std::function<bool(WordClass)> const& pred) const{
		return STRtoWSTR(ParseGenkeiAndFilter(WSTRtoSTR(sentence), pred));
	}

}