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

using RatingChunk = std::vector<typename SparseRatingMatrixBase_::const_iterator_range>;	


auto devide_rating_matrix(RatingMatrixPtr src, const uint n) ->std::tuple<RatingChunk, std::vector<RatingPtr>>
{
	auto&& src_users = src->getUsers();
	std::vector<RatingPtr> ratings;
	
	for (auto&& su : src_users){
		for(auto&& e : su) ratings.push_back(e);
	}
	sig::shuffle(ratings);
	
	RatingChunk chunks;
	const uint delta = ratings.size() / n;
	auto it = std::begin(ratings);
	for(uint i = 1; i < n; ++i, it += delta){
		chunks.push_back(boost::make_iterator_range(it, it + i*delta-1));
	}
	chunks.push_back(boost::make_iterator_range(it, std::end(ratings)));

	for(auto& chunk : chunks) boost::sort(chunk.begin(), chunk.end());
	
	return std::tmake_tuple(std::move(chunks), std::move(ratings));
}


template <class MODEL>
class CrossValidation;

class CrossValidationBase
{
protected:
	using const_iterator = SparseRatingMatrixBase_::const_iterator;
	using const_iterator_range = SparseRatingMatrixBase_::const_iterator_range;

protected:
	std::vector<RatingPtr> ratings_;
	RatingChunk rating_chunks_;

protected:
	CrossValidationBase() = default;
	virtual ~CrossValidationBase() = default;

	template <class CHUNC>
	auto join_chunc(CHUNC const& dataset, uint removed_index)
	{
		const_iterator_range result;

		for (uint i = 0; i < dataset.size(); ++i){
			if (i != removed_index) result = boost::join(result, dataseet[i]);
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