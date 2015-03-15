/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_DATA_FORMAT_H
#define SIGTM_DATA_FORMAT_H

#include "../sigtm.hpp"

#if SIG_USE_BOOST
#include <boost/bimap/bimap.hpp>
#include <boost/bimap/unordered_set_of.hpp>
#endif

namespace sigtm
{
struct Token;
//using TokenPtr = std::shared_ptr<Token const>;


/**
\brief
	@~japanese 文書中の１単語を表すトークン
	@~english a word-token in the document
*/
struct Token
{
	const uint self_id;
	const UserId user_id;
	const DocumentId doc_id;
	const WordId word_id;

	Token() = delete;
	Token(uint self_id, DocumentId d_id, WordId w_id) : self_id(self_id), doc_id(d_id), word_id(w_id), user_id(0){}
	Token(uint self_id, UserId u_id, DocumentId d_id, WordId w_id) : self_id(self_id), doc_id(d_id), word_id(w_id), user_id(u_id){}

	Token(Token const&) = default;
	Token& operator=(Token const& src){ Token tmp(src); std::swap(*this, tmp); return *this; }
};

/// トークン列
using TokenList = std::vector<Token>;


namespace impl
{
struct hash_C_WStrPtr
{
	size_t operator()(sig::C_WStrPtr const& x) const
	{
		return std::hash<std::wstring>()(*x);
	}
};

struct equal_C_WStrPtr
{
	bool operator()(sig::C_WStrPtr const& a, sig::C_WStrPtr const& b) const
	{
		return (*a) == (*b);
	}
};
}


/// 語彙集合
class WordSet
{
#if SIG_USE_BOOST
	using bimap = boost::bimaps::bimap< boost::bimaps::unordered_set_of<uint>, boost::bimaps::unordered_set_of<C_WStrPtr, impl::hash_C_WStrPtr, impl::equal_C_WStrPtr>>;
	using iterator = decltype(std::declval<bimap>().begin());
	using const_iterator = decltype(std::declval<bimap const>().begin());

	bimap id_word_;
#else
	using CWStr2IdMap = std::unordered_map<C_WStrPtr, Id, impl::hash_C_WStrPtr, impl::equal_C_WStrPtr>;
	using iterator = decltype(std::declval<std::vector<C_WStrPtr>>().begin());
	using const_iterator = decltype(std::declval<std::vector<C_WStrPtr> const>().begin());

	std::vector<C_WStrPtr> id_word_;
	CWStr2IdMap word_id_;
#endif

public:
	WordSet() = default;
	WordSet(WordSet const&) = delete;

#if SIG_USE_BOOST
	void emplace(WordId id, C_WStrPtr const& word){ id_word_.insert(bimap::value_type(id, word)); }
	void emplace(WordId id, std::wstring word){	id_word_.insert(bimap::value_type(id, std::make_shared<std::wstring>(word))); }
	
	//C_WStrPtr getWord(WordId id){ return id_word_.left[id]; }
	C_WStrPtr getWord(WordId id) const{ return id_word_.left.at(id); }
	C_WStrPtr getWord(const_iterator iter) const{ return iter->right; }
	C_WStrPtr getWord(const_iterator::value_type const& elem) const{ return elem.right; }

	//WordId getWordID(C_WStrPtr word){ return id_word_.right[word]; }
	WordId getWordID(C_WStrPtr const& word) const{ return id_word_.right.at(word); }
	WordId getWordID(const_iterator iter) const{ return iter->left; }
	WordId getWordID(const_iterator::value_type const& elem) const{ return elem.left; }

	bool hasElement(C_WStrPtr const& word) const{ return 0 < id_word_.right.count(word); }
	bool hasElement(WordId id) const{ return 0 < id_word_.left.count(id); }
#else
	void emplace(WordId id, C_WStrPtr word){
		id_word_.push_back(word);
		word_id_.emplace(word, id);
	}
	void emplace(WordId id, std::wstring word){
		auto wp = std::make_shared<std::wstring>(word);
		id_word_.push_back(wp);
		word_id_.emplace(wp, id);
	}

	C_WStrPtr getWord(WordId id) const{ return id_word_[id]; }
	C_WStrPtr getWord(const_iterator iter) const{ return *iter; }
	C_WStrPtr getWord(const_iterator::value_type const& elem) const{ return elem; }

	WordId getWordID(C_WStrPtr word) const{ return word_id_.at(word); }
	WordId getWordID(const_iterator iter) const{ return word_id_.at(*iter); }
	WordId getWordID(const_iterator::value_type const& elem) const{ return word_id_.at(elem); }

	bool hasElement(C_WStrPtr word) const{ return 0 < word_id_.count(word); }
	bool hasElement(WordId id) const{ return id < id_word_.size(); }
#endif

	auto begin() ->iterator{ return id_word_.begin(); }
	auto begin() const->const_iterator{ return id_word_.begin(); }

	auto end() ->iterator{ return id_word_.end(); }
	auto end() const->const_iterator{ return id_word_.end(); }

	uint size() const{ return id_word_.size(); }
};

}
#endif