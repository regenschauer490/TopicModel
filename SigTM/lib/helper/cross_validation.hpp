/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_CROSS_VALIDATION_H
#define SIGTM_CROSS_VALIDATION_H

#include "rating_matrix.hpp"
#include "SigUtil/lib/modify/shuffle.hpp"
#include <boost/range/join.hpp>
//#include <boost/range/adapter/slice.hpp>

namespace sigtm
{

class CTR;

using RatingChunk = std::vector<VectorU<RatingPtr>>;


auto devide_ratings(const uint n, RatingMatrixPtr src, bool at_user) ->RatingChunk
{
	auto&& src_vecs = at_user ? src->getUsers() : src->getItems();
	std::vector<RatingPtr> ratings;
	
	for (auto&& sv : src_vecs){
		for(auto&& e : sv) ratings.push_back(e);
	}
	sig::shuffle(ratings);
	
	RatingChunk chunks(n, VectorU<RatingPtr>(at_user ? src->userSize() : src->itemSize()));
	const uint delta = ratings.size() / n;

	for(uint i = 0; i <= n; ++i){
		for(uint j = i * delta, ed = (i+1) * delta, size = ed < ratings.size() ? ed : ratings.size(); j < size; ++j){
			chunks[i][at_user ? ratings[j]->user_id_ : ratings[j]->item_id_].push_back(ratings[j]);
		}
	}

	return chunks;
}

/
template <class MODEL>
class CrossValidation;

// validation for rating matrix
class CrossValidationBase
{
protected:
	using const_iterator = SparseRatingMatrixBase_::const_iterator;
	using const_iterator_range = SparseRatingMatrixBase_::const_iterator_range;

protected:
	RatingChunk rating_chunks_;

protected:
	CrossValidationBase() = default;
	virtual ~CrossValidationBase() = default;

	template <class CHUNC>
	auto join_chunc(CHUNC const& dataset, uint removed_index)
	{
		VectorU<const_iterator_range> result;

		for (uint i = 0; i < dataset.size(); ++i){
			if (i != removed_index){
				for(uint j = 0; j < dataset[i].size(); ++j){
					result[j] = boost::join(result[j], boost::make_iterator_range(std::begin(dataset[i][j]), std::end(dataset[i][j])));
				}
			}
		}

		return result;
	}

	template <class R, class F, class... ITs>
	auto run_parallel(F&& valid_func, uint n, ITs... iters) ->std::vector<R>
	{
		std::vector<R> result;
		std::vector<std::future< std::vector<R> >> task;

		for (uint i = 0; i < n; ++i) task.push_back(std::async(std::launch::async, valid_func, sig::impl::dereference_iterator(iters)...));

		for (auto& t : task){
			result.push_back(std::move(t.get()));
		}

		return result;
	}
};

}
#endif