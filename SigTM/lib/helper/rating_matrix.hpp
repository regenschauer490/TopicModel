/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_RATING_MATRIX_H
#define SIGTM_RATING_MATRIX_H

#include "../sigtm.hpp"
#include <boost/range/iterator_range.hpp>

namespace sigtm
{

using ItemId = uint;
template<class T> using VectorI = VectorD<T>;			// item

struct Rating_ : boost::noncopyable
{
	int value_;
	uint user_id_;
	uint item_id_;

public:
	Rating_(int value, uint user_id, uint item_id)
	: value_(value), user_id_(user_id), item_id_(item_id){}
};

using RatingPtr = std::shared_ptr<Rating_>;
using C_RatingPtr = std::shared_ptr<const Rating_>;


// user-item ratings for sparse matrix
class SparseRatingMatrixBase_ : boost::noncopyable
{
	using RatingContainer = std::vector<RatingPtr>;
	using UserRatings = VectorU<RatingContainer>;
	using ItemRatings = VectorI<RatingContainer>;
	
public:
	using iterator = RatingContainer::iterator;
	using const_iterator = RatingContainer::const_iterator;
	using iterator_range = boost::iterator_range<iterator>;
	using const_iterator_range = boost::iterator_range<const_iterator>;
	using rating_range = std::vector<iterator_range>;
	using const_rating_range = std::vector<const_iterator_range>;
	
protected:
	UserRatings user_;
	ItemRatings item_;

protected:
	SparseRatingMatrixBase_() = default;
	SparseRatingMatrixBase_(UserRatings u_src, ItemRatings i_src) : user_(std::move(u_src)), item_(std::move(i_src)){}

	virtual ~SparseRatingMatrixBase_() = default;
	
public:
	// return pair(begin, end)
	auto getUsers() const->const_rating_range{
		return sig::map([&](RatingContainer const& u){ return boost::make_iterator_range(std::begin(u), std::end(u)); }, user_);
	}

	auto getItems() const->const_rating_range{
		return sig::map([&](RatingContainer const& i){ return boost::make_iterator_range(std::begin(i), std::end(i)); }, item_);
	}
	
	/*auto getAllRatings() const{
		boost::range_iterator<RatingContainer> result;
		for(auto&& rs : user_) range = boost::join(range, rs);
		return result;
	}*/
	
	uint userSize() const{ return user_.size(); }
	uint itemSize() const{ return item_.size(); }
};

using RatingMatrixPtr = std::shared_ptr<SparseRatingMatrixBase_>;


// boolean user-item ratings(exist or not) for sparse matrix
class SparseBooleanMatrix : public SparseRatingMatrixBase_
{
	SparseBooleanMatrix(std::vector<std::vector<Id>> const& ratings, bool is_user_rating)
	{
		std::unordered_map<Id, std::vector<RatingPtr>> id_rating_map;
		uint row_id = 0;
		uint col_size = 0;

		if (is_user_rating) user_.resize(ratings.size());
		else item_.resize(ratings.size());

		for (auto const& row : ratings){
			for (Id id : row){
				if (is_user_rating){
					auto rating = std::make_shared<Rating_>(1, row_id, id);
					user_[row_id].push_back(rating);
					id_rating_map[id].push_back(rating);
				}
				else{
					auto rating = std::make_shared<Rating_>(1, id, row_id);
					item_[row_id].push_back(rating);
					id_rating_map[id].push_back(rating);
				}
				
				if(id > col_size ) col_size = id;
			}
			++row_id;
		}

		if (is_user_rating) item_.resize(col_size);
		else user_.resize(col_size);

		for (Id id = 0; id < col_size; ++id){
			if (is_user_rating){
				item_[id] = std::move(id_rating_map[id]);
			}
			else{
				user_[id] = std::move(id_rating_map[id]);
			}
		}
	}

	SparseBooleanMatrix(const_iterator_range const& ratings)
	{
		uint umax = 0, imax = 0;
		for(auto const& e : ratings){
			if(e->user_id_ > umax) umax = e->user_id_;
			if(e->item_id_ > imax) imax = e->item_id_;
		}
		
		user_.resize(umax);
		item_.resize(imax);
		
		for(auto const& e : ratings){
			user_[e->user_id_].push_back(e);
			item_[e->item_id_].push_back(e);
		}
	}
public:
	static auto makeInstance(std::vector<std::vector<Id>> const& ratings, bool is_user_rating) ->RatingMatrixPtr{
		return RatingMatrixPtr(new SparseBooleanMatrix(ratings, is_user_rating));
	}
	
	static auto makeInstance(const_iterator_range const& ratings) ->RatingMatrixPtr{
		return RatingMatrixPtr(new SparseBooleanMatrix(ratings));
	}
};

using BooleanMatrixPtr = std::shared_ptr<SparseBooleanMatrix>;

}
#endif