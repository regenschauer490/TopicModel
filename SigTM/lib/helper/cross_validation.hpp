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
auto devide_random(const uint n, SparseRatingMatrixBase_<T> const& src, bool is_user_test) ->RatingChunk<T>
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

template <class T>
auto devide_adjusted_random(const uint n, SparseRatingMatrixBase_<T> const& src, bool is_user_test) ->RatingChunk<T>
{
	auto&& src_vecs = is_user_test ? src.getUsers() : src.getItems();
	std::unordered_map<Id, std::vector<RatingPtr<T>>> ratings;

	if (is_user_test){
		for (auto&& sv : src_vecs){
			for (auto&& r : sv){
				if (ratings.count(r->item_id_)) ratings[r->item_id_].push_back(r);
				else ratings.emplace(r->item_id_, std::vector<RatingPtr<T>>{r});
			}
		}
	}
	else{

	}
	
	RatingChunk<T> chunks(n, std::vector<RatingContainer<T>>(is_user_test ? src.userSize() : src.itemSize()));

	sig::SimpleRandom<uint> random(0, n - 1, FixedRandom);
	uint ct = 0;

	if (is_user_test){
		for (auto& e : ratings){
			std::vector<RatingPtr<T>> rs = e.second;

			sig::shuffle(rs);
			
			if (rs.size() < n){
				sig::SimpleRandom<uint> random2(0, rs.size()-1, FixedRandom);

				for (uint i = 0; i < n; ++i){
					auto& r = rs[random2()];
					chunks[i][r->user_id_].push_back(r);
				}
				++ct;
			}
			else{
				uint rest = rs.size() - n;
				for (uint i = 0; i < n; ++i){
					chunks[i][rs[i]->user_id_].push_back(rs[i]);
				}
				for (uint j = n; j < rest; ++j){
					chunks[random()][rs[j]->user_id_].push_back(rs[j]);
				}
			}
		}
	}
	else{

	}

	std::cout << "less than n:" << ct << std::endl;

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
					for (auto&& e : boost::make_iterator_range(std::begin(dataset[i][j]), std::end(dataset[i][j]))) result[j].push_back(e);

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

		const uint cpu_core_num = std::thread::hardware_concurrency();
		const uint div = std::ceil(static_cast<double>(n) / cpu_core_num);

		for (uint d = 0; d < div; ++d){
			std::vector<std::future<R>> task;

			for (uint i = d * cpu_core_num, size = d != div-1 ? (d + 1) * cpu_core_num : n; i < size; ++i, sig::impl::increment_iterator(iters...)){
				task.push_back(std::async(std::launch::async, valid_func, sig::impl::dereference_iterator(iters)...));
			}

			for (auto& t : task){
				result.push_back(t.get());
			}
		}

		return result;
	}
};

}
#endif