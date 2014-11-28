/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_CROSS_VALIDATION_H
#define SIGTM_CROSS_VALIDATION_H

#include "rating_matrix.hpp"
#include "SigUtil/lib/modify/shuffle.hpp"
//#include <boost/range/adapter/slice.hpp>

namespace sigtm
{

class CTR;

template <class T>
using RatingChunk = std::vector<std::vector<RatingContainer<T>>>;


template <class T>
auto random_devide(const uint n, SparseRatingMatrixBase_<T> const& src, bool is_user_test) ->RatingChunk<T>
{
	auto&& src_vecs = is_user_test ? src.getUsers() : src.getItems();
	std::vector<RatingPtr<T>> ratings;
	
	for (auto&& sv : src_vecs){
		for(auto&& e : sv) ratings.push_back(e);
	}
	sig::shuffle(ratings);
	
	RatingChunk<T> chunks(n, std::vector<RatingContainer<T>>(is_user_test ? src.userSize() : src.itemSize()));
	const uint delta = ratings.size() / n;

	for(uint i = 0; i < n; ++i){
		for(uint j = i * delta, ed = (i+1) * delta, size = i != n-1 ? ed : ratings.size(); j < size; ++j){
			chunks[i][is_user_test ? ratings[j]->user_id_ : ratings[j]->item_id_].push_back(ratings[j]);
		}
	}

	return std::move(chunks);
}


template <class MODEL>
class CrossValidation;

// validation for rating matrix
template <class T>
class CrossValidationBase
{
protected:
	const uint div_num_;
	RatingChunk<T> rating_chunks_;

protected:
	CrossValidationBase(uint div_num) : div_num_(div_num){
		if (div_num < 2) std::terminate();
	};
	virtual ~CrossValidationBase() = default;

	template <class T, class R = typename RatingChunk<T>::value_type>
	auto join_chunc(RatingChunk<T> const& dataset, uint removed_index) ->R
	{
		R result(dataset[0].size());

		for (uint i = 0; i < dataset.size(); ++i){
			if (i != removed_index){
				for(uint j = 0; j < dataset[i].size(); ++j){
					for (auto&& e : boost::make_iterator_range(std::begin(dataset[i][j]), std::end(dataset[i][j]))) result[i].push_back(e);

					//jr[i] = boost::join(boost::any_cast<const_joined_range>(jr[j]), std::begin(dataset[i][j]), std::end(dataset[i][j]));
				}
			}
		}

		return result;
		//return sig::map([](boost::any const& e){ return boost::any_cast<const_joined_range>(e); }, jr);
	}

	template <class R, class F, class... ITs>
	auto run_parallel(F&& valid_func, uint n, ITs... iters) ->std::vector<R>
	{
		std::vector<R> result;
		std::vector<std::future<R>> task;

		for (uint i = 0; i < n; ++i) task.push_back(std::async(std::launch::async, valid_func, sig::impl::dereference_iterator(iters)...));

		for (auto& t : task){
			result.push_back(std::move(t.get()));
		}

		return result;
	}
};

}
#endif