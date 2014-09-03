/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_DATA_FORMAT_H
#define SIGTM_DATA_FORMAT_H

#include "../sigtm.hpp"

#if SIG_ENABLE_BOOST
#include <boost/bimap/bimap.hpp>
#include <boost/bimap/unordered_set_of.hpp>
#endif

namespace sigtm
{
struct Token;
//using TokenPtr = std::shared_ptr<Token const>;


/* ある単語を表すトークン */
struct Token
{
	const uint self_id;
	const UserId user_id;
	const DocumentId doc_id;
	const WordId word_id;

	Token() = delete;
	Token(uint self_id, DocumentId d_id, WordId w_id) : self_id(self_id), doc_id(d_id), word_id(w_id), user_id(0){}
	Token(uint self_id, UserId u_id, DocumentId d_id, WordId w_id) : self_id(self_id), doc_id(d_id), word_id(w_id), user_id(u_id){}
};

/* トークン列 */
using TokenList = std::vector<Token>;


#if SIG_ENABLE_BOOST
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
#endif


/* 単語集合 */
class WordSet
{
#if SIG_ENABLE_BOOST
	using bimap = boost::bimaps::bimap< boost::bimaps::unordered_set_of<uint>, boost::bimaps::unordered_set_of<C_WStrPtr, hash_C_WStrPtr, equal_C_WStrPtr>>;
	using iterator = decltype(std::declval<bimap>().begin());
	using const_iterator = decltype(std::declval<bimap const>().begin());

	bimap id_word_;
#else
	using iterator = decltype(std::declval<std::vector<C_WStrPtr>>().begin());
	using const_iterator = decltype(std::declval<std::vector<C_WStrPtr> const>().begin());

	std::vector<C_WStrPtr> id_word_;
	std::unordered_map<C_WStrPtr, uint> word_id_;
#endif

public:
	WordSet() = default;
	WordSet(WordSet const&) = delete;

#if SIG_ENABLE_BOOST
	void emplace(WordId id, C_WStrPtr word){ id_word_.insert(bimap::value_type(id, word)); }
	
	//C_WStrPtr getWord(WordId id){ return id_word_.left[id]; }
	C_WStrPtr getWord(WordId id) const{ return id_word_.left.at(id); }
	C_WStrPtr getWord(const_iterator iter) const{ return iter->right; }
	C_WStrPtr getWord(const_iterator::value_type const& elem) const{ return elem.right; }

	//WordId getWordID(C_WStrPtr word){ return id_word_.right[word]; }
	WordId getWordID(C_WStrPtr word) const{ return id_word_.right.at(word); }
	WordId getWordID(const_iterator iter) const{ return iter->left; }
	WordId getWordID(const_iterator::value_type const& elem) const{ return elem.left; }

	bool hasElement(C_WStrPtr word) const{ return id_word_.right.count(word); }
	bool hasElement(WordId id) const{ return id_word_.left.count(id); }
#else
	void emplace(WordId id, C_WStrPtr word){
		id_word_.push_back(word);
		word_id_.emplace(word, id);
	}

	C_WStrPtr getWord(WordId id) const{ return id_word_[id]; }
	C_WStrPtr getWord(const_iterator iter) const{ return *iter; }
	C_WStrPtr getWord(const_iterator::value_type const& elem) const{ return elem; }

	WordId getWordID(C_WStrPtr word) const{ return word_id_[word]; }
	WordId getWordID(const_iterator iter) const{ return word_id_[*iter]; }
	WordId getWordID(const_iterator::value_type const& elem) const{ return word_id_[elem]; }

	bool hasElement(C_WStrPtr word) const{ return word_id_.count(word); }
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