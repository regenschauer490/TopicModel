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

template<class T> using VectorD = std::vector<T>;	// document
template<class T> using VectorK = std::vector<T>;	// topic
template<class T> using VectorV = std::vector<T>;	// word
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

	void calcTermScore(MatrixKV<double> const& beta, MatrixKV<double>& dest) const;

	auto getTopWords(VectorV<double> const& dist, uint num, WordSet const& words) const->std::vector<std::tuple<std::wstring, double>>;

	template <class CC>
	void printWord(CC const& data, WordSet const& words, maybe<uint> top_num, maybe<FilepassString> save_pass, bool detail) const;

	template <class CC>
	void printTopic(CC const& data, maybe<FilepassString> save_pass) const;

public:
	virtual ~LDA(){}
	
	// モデルの学習を行う
	virtual void learn(uint iteration_num) = 0;
		
	// コンソールに出力
	virtual void print(Distribution target) const = 0;

	// ファイルに出力
	// save_folder: 保存先のフォルダのパス
	// detail: 詳細なデータも全て出力するか
	virtual void save(Distribution target, FilepassString save_folder, bool detail = false) const = 0;

	//ドキュメントのトピック分布 [doc][topic]
	virtual auto getTopicDistribution() const->MatrixDK<double> = 0;
	virtual auto getTopicDistribution(DocumentId d_id) const->VectorK<double> = 0;

	//トピックの単語分布 [topic][word]
	virtual auto getWordDistribution() const->MatrixKV<double> = 0;
	virtual auto getWordDistribution(TopicId k_id) const->VectorV<double> = 0;

	//トピックを強調する語スコア [topic][word]
	virtual auto getTermScoreOfTopic() const->MatrixKV<double> = 0;
	virtual auto getTermScoreOfTopic(TopicId t_id) const->VectorV<double> = 0;

	//ドキュメントのThetaとTermScoreの積 [ranking]<word_id,score>
	virtual auto getTermScoreOfDocument(DocumentId d_id) const->std::vector< std::tuple<WordId, double> > = 0;

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
	virtual auto getEta() const->VectorV<double> = 0;
};


inline void LDA::calcTermScore(MatrixKV<double> const& beta, MatrixKV<double>& dest) const
{
	const uint K = beta.size();
	const uint V = std::begin(beta)->size();

	const auto Task_div = [&](uint const begin, uint const end){
		std::vector< std::vector<double> > ts(K);

		for (uint _w = begin, i = 0; _w < end; ++_w, ++i){
			double ip = 1;
			for (uint k2 = 0; k2 < K; ++k2){
				ip *= beta[k2][_w];
			}
			ip = pow(ip, 1.0 / K);

			for (uint k = 0; k < K; ++k){
				ts[k].push_back(beta[k][_w] * log(beta[k][_w] / ip));
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

template <class CC>
void LDA::printWord(CC const& data, WordSet const& words, maybe<uint> top_num, maybe<FilepassString> save_pass, bool detail) const
{
	auto Output = [](std::wostream& ofs, std::vector<std::tuple<std::wstring, double>> const& data, maybe<std::wstring> header)
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

		if (save_pass){
			Output(ofs, rank_words, L"id:" + std::to_wstring(i));
		}
		else{
			Output(std::wcout, rank_words, L"id:" + std::to_wstring(i));
		}
	}
	, 1, data);

	if (save_pass && detail){
		sig::for_each([&](int i, VectorV<double> const& d)
		{
			std::wofstream ofs2(sig::fromJust(save_pass) + sig::to_fpstring(i) + SIG_STR_TO_FPSTR(".txt"));
			for (auto const& e : d) ofs2 << e << L' ' << *words.getWord(i-1) << std::endl;
		}
		, 1, data);
	}
}

template <class CC>
void LDA::printTopic(CC const& data, maybe<FilepassString> save_pass) const
{
	auto Output = [](std::wostream& ofs, VectorK<double> data, maybe<std::wstring> header)
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
		if (save_pass){
			Output(ofs, d, L"id:" + std::to_wstring(i));
		}
		else{
			Output(std::wcout, d, L"id:" + std::to_wstring(i));
		}
	}
	, 1, data);
}


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
//		return std::static_pointer_cast<MrLDA>(lda)->compare<Select>(id1, id2);
	default :
		assert(false);
	}
}

}
#endif
