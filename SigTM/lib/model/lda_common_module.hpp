/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_LDA_COMMON_MODULE_HPP
#define SIGTM_LDA_COMMON_MODULE_HPP

#include "../sigtm.hpp"
#include "lda_interface.hpp"
#include "SigUtil/lib/modify/sort.hpp"
#include "SigUtil/lib/tools/random.hpp"
#include "SigUtil/lib/calculation/basic_statistics.hpp"
#include <future>

#include "../helper/document_loader.hpp"


namespace sigtm
{
namespace impl
{
const uint cpu_core_num = std::thread::hardware_concurrency();

class LDA_Module
{
protected:
	template <class MatrixKVd1, class MatrixKVd2>
	void calcTermScore(MatrixKVd1 const& phi, MatrixKVd2& dest) const;

	template <class VectorVd>
	auto getTopWords(VectorVd const& dist, uint num, WordSet const& words) const->std::vector<std::tuple<std::wstring, double>>;

	template <class VectorKd, class MatrixKVd>
	auto getTermScoreOfDocument(VectorKd const& theta, MatrixKVd const& tscore) const->std::vector< std::tuple<WordId, double>>;

	// data[class][word], names[class]
	template <class CC>
	void printWord(CC const& data, std::vector<FilepassString> const& names, WordSet const& words, Maybe<uint> top_num, Maybe<FilepassString> save_pass) const;

	// data[class][topic], names[class]
	template <class CC>
	void printTopic(CC const& data, std::vector<FilepassString> const& names, Maybe<FilepassString> save_pass) const;

	double calcLogLikelihood(TokenList const& tokens, MatrixDK<double> const& theta, MatrixKV<double> const& phi) const;
};


template <class MatrixKVd1, class MatrixKVd2>
void LDA_Module::calcTermScore(MatrixKVd1 const& phi, MatrixKVd2& dest) const
{
	const uint K = phi.size();
	const uint V = std::begin(phi)->size();

	const auto task_div = [&](uint const begin, uint const end){
		std::vector< std::vector<double> > ts(K);

		for (uint _w = begin, i = 0; _w < end; ++_w, ++i){
			double ip = std::pow(2, K);
			for (uint k2 = 0; k2 < K; ++k2){
				ip *= phi[k2][_w];
			}
			ip = pow(ip, 1.0 / K);

			for (uint k = 0; k < K; ++k){
				ts[k].push_back(phi[k][_w] * log(phi[k][_w] / ip));
				if (ts[k][i] < std::numeric_limits<double>::epsilon()) ts[k][i] = 0.0;
			}
		}
		return std::move(ts);
	};

	uint const div_size = V / cpu_core_num; //ThreadNum;
	std::vector<std::future< std::vector<std::vector<double>> >> task;

	for (uint i = 0, w = 0, we = div_size; i<cpu_core_num + 1; ++i, w += div_size, we += div_size){
		if (we > V) we = V;
		task.push_back(std::async(std::launch::async, task_div, w, we));
	}

	WordId w = 0;
	for (auto& t : task){
		auto vec = t.get();
		for (uint i = 0, size = vec[0].size(); i<size; ++i, ++w){
			for (TopicId k = 0; k<K; ++k) dest[k][w] = vec[k][i];
		}
	}
}

template <class VectorVd>
auto LDA_Module::getTopWords(VectorVd const& dist, uint num, WordSet const& words) const->std::vector<std::tuple<std::wstring, double>>
{
	std::vector< std::tuple<std::wstring, double> > result;
	std::vector< std::tuple<WordId, double> > tmp;

	auto sorted = sig::sort_with_index(dist, std::greater<double>());
	auto sorted_dist = std::get<0>(sorted);
	auto sorted_wid = std::get<1>(sorted);

	for (uint i = 0; i < num; ++i){
		result.push_back(std::make_tuple(*words.getWord(sorted_wid[i]), sorted_dist[i]));
	}
	return std::move(result);
}

template <class VectorKd, class MatrixKVd>
auto LDA_Module::getTermScoreOfDocument(VectorKd const& theta, MatrixKVd const& tscore) const->std::vector< std::tuple<WordId, double>>
{
	VectorV<double> tmp(std::begin(tscore)->size(), 0.0);
	TopicId t = 0;

	for (auto d1 = theta.begin(), d1end = theta.end(); d1 != d1end; ++d1, ++t){
		WordId w = 0;
		for (auto d2 = tscore[t].begin(), d2end = tscore[t].end(); d2 != d2end; ++d2, ++w){
			tmp[w] += ((*d1) * (*d2));
		}
	}

	auto sorted = sig::sort_with_index(tmp, std::less<double>()); //std::tuple<std::vector<double>, std::vector<uint>>
	return sig::zipWith([](WordId w, double d){ return std::make_tuple(w, d); }, std::get<1>(sorted), std::get<0>(sorted)); //sig::zip(std::get<1>(sorted), std::get<0>(sorted));
}

template <class CC>
void LDA_Module::printWord(CC const& data, std::vector<FilepassString> const& names, WordSet const& words, Maybe<uint> top_num, Maybe<FilepassString> save_pass) const
{
	auto Output = [](std::wostream& ofs, std::vector<std::tuple<std::wstring, double>> const& data, Maybe<FilepassString> header)
	{
		if (header) ofs << sig::fromJust(header) << std::endl;
		for (auto const& e : data){
			ofs << std::get<0>(e) << L' ' << std::get<1>(e) << std::endl;
		}
		ofs << std::endl;
	};

	auto ofs = save_pass ? std::wofstream(sig::fromJust(save_pass) + SIG_TO_FPSTR(".txt")) : std::wofstream(SIG_TO_FPSTR(""));

	// 各クラス(ex.トピック)のスコア上位top_num個の単語を出力
	sig::for_each([&](int i, VectorV<double> const& wscore)
	{
		auto rank_words = top_num ? getTopWords(wscore, sig::fromJust(top_num), words) : getTopWords(wscore, wscore.size(), words);
		auto header = L"class:" + (names.empty() ? std::to_wstring(i) : names[i - 1]);

		if (save_pass){
			Output(ofs, rank_words, header);
		}
		else{
			Output(std::wcout, rank_words, header);
		}
	}
	, 1, data);
	/*
	if (save_pass && detail){
	// 全単語を出力
	sig::for_each([&](int i, VectorV<double> const& d)
	{
	auto header = names.empty() ? L"class:" + std::to_wstring(i) : names[i-1];
	std::wofstream ofs2(sig::fromJust(save_pass) + header + SIG_TO_FPSTR(".txt"));
	for (auto const& e : d) ofs2 << e << L' ' << *words.getWord(i-1) << std::endl;
	}
	, 1, data);
	}
	*/
}

template <class CC>
void LDA_Module::printTopic(CC const& data, std::vector<FilepassString> const& names, Maybe<FilepassString> save_pass) const
{
	auto Output = [](std::wostream& ofs, VectorK<double> data, Maybe<FilepassString> header)
	{
		if (header) ofs << sig::fromJust(header) << std::endl;
		for (auto const& e : data){
			ofs << e << std::endl;
		}
		ofs << std::endl;
	};

	auto ofs = save_pass ? std::wofstream(sig::fromJust(save_pass) + SIG_TO_FPSTR(".txt")) : std::wofstream(SIG_TO_FPSTR(""));

	// 各クラス(ex.ドキュメント)のトピック分布を出力
	sig::for_each([&](int i, VectorK<double> const& tscore)
	{
		auto header = L"id:" + (names.empty() ? std::to_wstring(i) : names[i - 1]);
		if (save_pass){
			Output(ofs, tscore, header);
		}
		else{
			Output(std::wcout, tscore, header);
		}
	}
	, 1, data);
}

inline double LDA_Module::calcLogLikelihood(TokenList const& tokens, MatrixDK<double> const& theta, MatrixKV<double> const& phi) const
{
	double log_likelihood = 0;
	const uint K = phi.size();

	for (auto const& token : tokens){
		const auto& theta_d = theta[token.doc_id];
		uint w = token.word_id;
		double tmp = 0;
		for (uint k = 0; k < K; ++k){
			tmp += theta_d[k] * phi[k][w];
		}
		log_likelihood += std::log(tmp);
	}

	return log_likelihood;
}

}	// namespace impl

}	// namespace sigtm

#endif