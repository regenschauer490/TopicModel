/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#if USE_SIGNLP

#include "mecab_wrapper.h"

namespace signlp{
	using sig::wstr_to_str;
	using sig::str_to_wstr;
	using sig::split;

#define PARSE_WSTRING_VER(parse_func) \
	std::vector< std::tuple<std::wstring, WordClass> > result; \
	\
	auto tmp = parse_func(wstr_to_str(sentence)); \
	std::transform(tmp.begin(), tmp.end(), std::back_inserter(result), [](std::tuple<std::string, WordClass> const& e){ \
		return std::make_tuple(str_to_wstr(std::get<0>(e)), std::get<1>(e)); \
	}); \
	\
	return std::move(result);


	volatile void MecabWrapper::ParseImpl(std::string const& src, std::string& dest) const
	{
		TaggerPtr tagger(model_->createTagger());
		LatticePtr lattice(model_->createLattice());
		/*	if(!tagger || !lattice){
		getchar();
		}*/
		lattice->set_sentence(src.c_str());

		tagger->parse(lattice.get());

		dest.assign(lattice->toString());
	}

	std::vector<std::string> MecabWrapper::parseSimple(std::string const& sentence) const
	{
		std::vector<std::string> result;
		if (enable_warning && sentence.empty()){ std::cout << "sentense is empty" << std::endl; return result; }

		std::string parse;
		ParseImpl(sentence, parse);
		auto wlist = split(parse, "\n");
		for (auto& w : wlist){
			auto tmp = split(w, "\t");
			result.push_back(tmp[0]);
		}

		return std::move(result);
	}
	std::vector<std::wstring> MecabWrapper::parseSimple(std::wstring const& sentence) const{
		return str_to_wstr(parseSimple(wstr_to_str(sentence)));
	}

	std::vector< std::tuple<std::string, WordClass> > MecabWrapper::parseSimpleWithWC(std::string const& sentence) const
	{
		std::vector< std::tuple<std::string, WordClass> > result;
		if (enable_warning && sentence.empty()){ std::cout << "sentense is empty" << std::endl; return result; }

		std::string parse;
		ParseImpl(sentence, parse);
		auto wlist = split(parse, "\n");
		for (auto& w : wlist){
			auto tmp = split(w, "\t");
			if (tmp.size() < 2) continue;
			auto tmp2 = split(tmp[1], ",");
			if (tmp2.size() < 7) continue;
			result.push_back(std::make_tuple(tmp[0], StrToWC(tmp2[0])));
		}

		return std::move(result);
	}
	std::vector< std::tuple<std::wstring, WordClass> > MecabWrapper::parseSimpleWithWC(std::wstring const& sentence) const
	{
		PARSE_WSTRING_VER(parseSimpleWithWC);
	}

	std::vector<std::string> MecabWrapper::parseGenkei(std::string const& sentence, bool skip) const
	{
		std::vector<std::string> result;
		if (enable_warning && sentence.empty()){ std::cout << "sentense is empty" << std::endl; return result; }

		std::string parse;
		ParseImpl(sentence, parse);
		auto wlist = split(parse, "\n");

		for (auto& w : wlist){
			auto tmp = split(w, "\t");
			if (tmp.size() < 2) continue;
			auto tmp2 = split(tmp[1], ",");
			if (tmp2.size() < 7) continue;
			if (tmp2[6] == "*"){
				if (!skip) result.push_back(tmp[0]);
			}
			else result.push_back(tmp2[6]);
		}

		return std::move(result);
	}
	std::vector<std::wstring> MecabWrapper::parseGenkei(std::wstring const& sentence, bool skip) const{
		return str_to_wstr(parseGenkei(wstr_to_str(sentence)));
	}

	std::vector< std::tuple<std::string, WordClass> > MecabWrapper::parseGenkeiWithWC(std::string const& sentence, bool skip) const
	{
		std::vector< std::tuple<std::string, WordClass> > result;
		if (enable_warning && sentence.empty()){ std::cout << "sentense is empty" << std::endl; return result; }

		std::string parse;
		ParseImpl(sentence, parse);

		auto wlist = split(parse, "\n");
		for (auto& w : wlist){
			auto tmp = split(w, "\t");
			if (tmp.size() < 2) continue;
			auto tmp2 = split(tmp[1], ",");
			if (tmp2.size() < 7) continue;
			if (tmp2[6] == "*"){
				if (!skip) result.push_back(std::make_tuple(tmp[0], StrToWC(tmp2[0])));
			}
			else result.push_back(std::make_tuple(tmp2[6], StrToWC(tmp2[0])));
		}

		return std::move(result);
	}
	std::vector< std::tuple<std::wstring, WordClass> > MecabWrapper::parseGenkeiWithWC(std::wstring const& sentence, bool skip) const
	{
		PARSE_WSTRING_VER(parseGenkeiWithWC);
	}

	std::vector<std::string> MecabWrapper::parseSimpleThroughFilter(std::string const& sentence, std::function<bool(WordClass)> const& pred) const
	{
		auto parse = parseSimpleWithWC(sentence);

		std::vector<std::string> result;
		for (const auto& w : parse){
			if (pred(std::get<1>(w))) result.push_back(std::get<0>(w));
		}

		return std::move(result);
	}
	std::vector<std::wstring> MecabWrapper::parseSimpleThroughFilter(std::wstring const& sentence, std::function<bool(WordClass)> const& pred) const{
		return str_to_wstr(parseSimpleThroughFilter(wstr_to_str(sentence), pred)); 
	}

	std::vector<std::string> MecabWrapper::parseGenkeiThroughFilter(std::string const& sentence, std::function<bool(WordClass)> const& pred) const
	{
		auto parse = parseGenkeiWithWC(sentence);

		std::vector<std::string> result;
		for (const auto& w : parse){
			if (pred(std::get<1>(w))) result.push_back(std::get<0>(w));
		}

		return std::move(result);
	}
	std::vector<std::wstring> MecabWrapper::parseGenkeiThroughFilter(std::wstring const& sentence, std::function<bool(WordClass)> const& pred) const{
		return str_to_wstr(parseGenkeiThroughFilter(wstr_to_str(sentence), pred));
	}

}
#endif