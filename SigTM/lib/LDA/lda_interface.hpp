/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_LDA_INTERFACE_HPP
#define SIGTM_LDA_INTERFACE_HPP

#include "../sigtm.hpp"
#include "../helper/compare_method.hpp"
#include "../helper/data_format.hpp"
#include "SigUtil/lib/modify.hpp"
#include <future>

namespace sigtm
{

class LDA;
using LDAPtr = std::shared_ptr<LDA>;

template<class T> using VectorT = std::vector<T>;	// all token
template<class T> using VectorD = std::vector<T>;	// document
template<class T> using VectorK = std::vector<T>;	// topic
template<class T> using VectorV = std::vector<T>;	// word
template<class T> using MatrixTK = VectorT<VectorK<T>>;	// token - topic
template<class T> using MatrixDK = VectorD<VectorK<T>>;	// document - topic
template<class T> using MatrixVK = VectorV<VectorK<T>>;	// word - topic
template<class T> using MatrixKV = VectorK<VectorV<T>>;	// topic - word
/*
template<class T> using SVectorD = std::shared_ptr<VectorD<T>>;
template<class T> using SVectorK = std::shared_ptr<VectorK<T>>;
template<class T> using SVectorV = std::shared_ptr<VectorV<T>>;
template<class T> using SMatrixDK = SVectorD<SVectorK<T>>;
template<class T> using SMatrixVK = SVectorV<SVectorK<T>>;
template<class T> using SMatrixKV = SVectorK<SVectorV<T>>;
*/

#define SIG_INIT_VECTOR(type, index_name, value)\
	Vector ## index_name ## <type>(index_name ## _, value)

#define SIG_INIT_MATRIX(type, index_name1, index_name2, value)\
	Matrix ## index_name1 ## index_name2 ## <type>(index_name1 ## _, SIG_INIT_VECTOR(type, index_name2, value))

const double default_alpha_base = 50;
const double default_beta = 0.1;

// LDAインタフェース
class LDA
{
public:
	enum class DynamicType{ GIBBS, MRLDA };		// 実装次第追加
	virtual DynamicType getDynamicType() const = 0;

public:
	// LDAで得られる確率分布やベクトル
	enum class Distribution{ DOCUMENT, TOPIC, TERM_SCORE };

	SIG_MakeCompareInnerClass(LDA);

protected:
	// method chain 生成
	SIG_MakeDist2CmpMapBase;
	SIG_MakeDist2CmpMap(Distribution::DOCUMENT, LDA::CmpD<std::function< VectorD<double>(DocumentId) >>);
	SIG_MakeDist2CmpMap(Distribution::TOPIC, LDA::CmpD<std::function< VectorK<double>(TopicId) >>);
	SIG_MakeDist2CmpMap(Distribution::TERM_SCORE, LDA::CmpV<std::function< VectorK<double>(TopicId) >>);
	//SIG_MakeDist2CmpMap(Distribution::DOC_TERM, LDA::CmpV);
	
	template <LDA::Distribution Select> friend auto compare(LDAPtr lda, Id id1, Id id2) ->typename Map2Cmp<Select>::type;

protected:
	LDA() = default;
	LDA(LDA const&) = delete;
	LDA(LDA&&) = delete;

	void calcTermScore(MatrixKV<double> const& phi, MatrixKV<double>& dest) const;

	auto getTopWords(VectorV<double> const& dist, uint num, WordSet const& words) const->std::vector<std::tuple<std::wstring, double>>;

	auto getTermScoreOfDocument(DocumentId d_id) const->std::vector< std::tuple<WordId, double>>;

	template <class CC>
	void printWord(CC const& data, std::vector<FilepassString> const& names, WordSet const& words, maybe<uint> top_num, maybe<FilepassString> save_pass, bool detail) const;

	template <class CC>
	void printTopic(CC const& data, std::vector<FilepassString> const& names, maybe<FilepassString> save_pass) const;

	// 確率分布同士の類似度を測る(メソッドチェーンな感じに使用)
	template <Distribution Select>
	auto compareDefault(Id id1, Id id2, uint D, uint K) const->typename Map2Cmp<Select>::type
	{
		return Select == Distribution::DOCUMENT
			? typename Map2Cmp<Select>::type(id1, id2, [this](DocumentId id){ return this->getTheta(id); }, id1 < D && id2 < D ? true : false)
			: Select == Distribution::TOPIC
				? typename Map2Cmp<Select>::type(id1, id2, [this](TopicId id){ return this->getPhi(id); }, id1 < K && id2 < K ? true : false)
				: Select == Distribution::TERM_SCORE
					? typename Map2Cmp<Select>::type(id1, id2, [this](TopicId id){ return this->getTermScore(id); }, id1 < K && id2 < K ? true : false)
					: typename Map2Cmp<Select>::type(id1, id2, [](TopicId id){ return std::vector<double>(); }, false);
	}
	
public:
	virtual ~LDA(){}
	
	// モデルの学習を行う
	virtual void train(uint iteration_num) = 0;

	virtual void train(uint iteration_num, std::function<void(LDA const*)> callback) = 0;
		
	// コンソールに出力
	virtual void print(Distribution target) const = 0;

	// ファイルに出力
	// save_folder: 保存先のフォルダのパス
	// detail: 詳細なデータも全て出力するか
	virtual void save(Distribution target, FilepassString save_folder, bool detail = false) const = 0;

	//ドキュメントのトピック分布 [doc][topic]
	virtual auto getTheta() const->MatrixDK<double> = 0;
	virtual auto getTheta(DocumentId d_id) const->VectorK<double> = 0;

	//トピックの単語分布 [topic][word]
	virtual auto getPhi() const->MatrixKV<double> = 0;
	virtual auto getPhi(TopicId k_id) const->VectorV<double> = 0;

	//トピックを強調する単語スコア [topic][word]
	virtual auto getTermScore() const->MatrixKV<double> = 0;
	virtual auto getTermScore(TopicId t_id) const->VectorV<double> = 0;
	
	// 指定トピックの上位return_word_num個の、語彙とスコアを返す
	// [topic][ranking]<vocab, score>
	virtual auto getWordOfTopic(Distribution target, uint return_word_num) const->VectorK< std::vector< std::tuple<std::wstring, double> > > = 0;
	// [ranking]<vocab, score>
	virtual auto getWordOfTopic(Distribution target, uint return_word_num, TopicId k_id) const->std::vector< std::tuple<std::wstring, double> > = 0;

	// 指定ドキュメントの上位return_word_num個の、語彙とスコアを返す
	// [doc][ranking]<vocab, score>
	virtual auto getWordOfDocument(uint return_word_num) const->VectorD< std::vector< std::tuple<std::wstring, double> > > = 0;
	//[ranking]<vocab, score>
	virtual auto getWordOfDocument(uint return_word_num, DocumentId d_id) const->std::vector< std::tuple<std::wstring, double> > = 0;

	virtual uint getDocumentNum() const = 0;
	virtual uint getTopicNum() const = 0;
	virtual uint getWordNum() const = 0;

	// get hyper-parameter of topic distribution
	virtual auto getAlpha() const->VectorK<double> = 0;
	// get hyper-parameter of word distribution
	virtual auto getBeta() const->MatrixKV<double> = 0;

	virtual double getLogLikelihood() const = 0;

	virtual double getPerplexity() const = 0;
};


inline void LDA::calcTermScore(MatrixKV<double> const& phi, MatrixKV<double>& dest) const
{
	const uint K = phi.size();
	const uint V = std::begin(phi)->size();

	const auto Task_div = [&](uint const begin, uint const end){
		std::vector< std::vector<double> > ts(K);

		for (uint _w = begin, i = 0; _w < end; ++_w, ++i){
			double ip = 1;
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

	uint const div_size = V / ThreadNum;
	std::vector<std::future< std::vector<std::vector<double>> >> task;

	for (uint i = 0, w = 0, we = div_size; i<ThreadNum + 1; ++i, w += div_size, we += div_size){
		if (we > V) we = V;
		task.push_back(std::async(std::launch::async, Task_div, w, we));
	}

	WordId w = 0;
	for (auto& t : task){
		auto vec = t.get();
		for (uint i = 0, size = vec[0].size(); i<size; ++i, ++w){
			for (TopicId k = 0; k<K; ++k) dest[k][w] = vec[k][i];
		}
	}
}

inline auto LDA::getTopWords(VectorV<double> const& dist, uint num, WordSet const& words) const->std::vector<std::tuple<std::wstring, double>>
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

inline auto LDA::getTermScoreOfDocument(DocumentId d_id) const->std::vector< std::tuple<WordId, double>>
{
	const auto theta = getTheta(d_id);
	const auto tscore = getTermScore();

	VectorV<double> tmp(std::begin(tscore)->size(), 0.0);
	TopicId t = 0;

	for (auto d1 = theta.begin(), d1end = theta.end(); d1 != d1end; ++d1, ++t){
		WordId w = 0;
		for (auto d2 = tscore[t].begin(), d2end = tscore[t].end(); d2 != d2end; ++d2, ++w){
			tmp[w] += ((*d1) * (*d2));
		}
	}

	auto sorted = sig::sort_with_index(tmp); //std::tuple<std::vector<double>, std::vector<uint>>
	return sig::zipWith([](WordId w, double d){ return std::make_tuple(w, d); }, std::get<1>(sorted), std::get<0>(sorted)); //sig::zip(std::get<1>(sorted), std::get<0>(sorted));
}

template <class CC>
void LDA::printWord(CC const& data, std::vector<FilepassString> const& names, WordSet const& words, maybe<uint> top_num, maybe<FilepassString> save_pass, bool detail) const
{
	auto Output = [](std::wostream& ofs, std::vector<std::tuple<std::wstring, double>> const& data, maybe<FilepassString> header)
	{
		if(header) ofs << sig::fromJust(header) << std::endl;
		for (auto const& e : data){
			ofs << std::get<0>(e) << L' ' << std::get<1>(e) << std::endl;
		}
		ofs << std::endl;
	};

	auto ofs = save_pass ? std::wofstream(sig::fromJust(save_pass) + SIG_STR_TO_FPSTR(".txt")) : std::wofstream(SIG_STR_TO_FPSTR(""));

	sig::for_each([&](int i, VectorV<double> const& d)
	{
		auto rank_words = top_num ? getTopWords(d, sig::fromJust(top_num), words) : getTopWords(d, d.size(), words);
		auto header = names.empty() ? L"id:" + std::to_wstring(i) : names[i-1];

		if (save_pass){
			Output(ofs, rank_words, header);
		}
		else{
			Output(std::wcout, rank_words, header);
		}
	}
	, 1, data);

	if (save_pass && detail){
		sig::for_each([&](int i, VectorV<double> const& d)
		{
			auto header = names.empty() ? L"id:" + std::to_wstring(i) : names[i-1];
			std::wofstream ofs2(sig::fromJust(save_pass) + header + SIG_STR_TO_FPSTR(".txt"));
			for (auto const& e : d) ofs2 << e << L' ' << *words.getWord(i-1) << std::endl;
		}
		, 1, data);
	}
}

template <class CC>
void LDA::printTopic(CC const& data, std::vector<FilepassString> const& names, maybe<FilepassString> save_pass) const
{
	auto Output = [](std::wostream& ofs, VectorK<double> data, maybe<FilepassString> header)
	{
		if (header) ofs << sig::fromJust(header) << std::endl;
		for (auto const& e : data){
			ofs << e << std::endl;
		}
		ofs << std::endl;
	};

	auto ofs = save_pass ? std::wofstream(sig::fromJust(save_pass) + SIG_STR_TO_FPSTR(".txt")) : std::wofstream(SIG_STR_TO_FPSTR(""));

	sig::for_each([&](int i, VectorK<double> const& d)
	{
		auto header = names.empty() ? L"id:" + std::to_wstring(i) : names[i-1];
		if (save_pass){
			Output(ofs, d, header);
		}
		else{
			Output(std::wcout, d, header);
		}
	}
	, 1, data);
}


const std::function<void(LDA const*)> null_lda_callback = [](LDA const*){};

class LDA_Gibbs;
class MrLDA;

// 確率分布同士の類似度を測る(メソッドチェーンな感じに使用)
template <LDA::Distribution Select>
auto compare(LDAPtr lda, Id id1, Id id2) ->typename LDA::Map2Cmp<Select>::type
{
	switch (lda->getDynamicType()){
	case LDA::DynamicType::GIBBS :
		return std::static_pointer_cast<LDA_Gibbs>(lda)->compare<Select>(id1, id2);
	case LDA::DynamicType::MRLDA :
		return std::static_pointer_cast<MrLDA>(lda)->compare<Select>(id1, id2);
	default :
		assert(false);
	}
}

}
#endif
