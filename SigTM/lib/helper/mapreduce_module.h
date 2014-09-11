/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_MAPREDUCE_MODULE_H
#define  SIGTM_MAPREDUCE_MODULE_H

#include "../model/lda_interface.hpp"
#include "../../external/mapreduce/include/mapreduce.hpp"

namespace sigtm
{
class MrLDA;

namespace mrlda{

// map時にkeyとして渡されるオブジェクト
struct MapValue
{
	uint knum_;
	uint vnum_;
	VectorV<uint> const* word_ct_;
	VectorK<double> const* alpha_;
	MatrixKV<double> const* phi_;
	VectorK<double>* gamma_;		// gammaは各mapperで更新

	MapValue() = default;
	//MapValue(uint topic_num, uint word_num) : knum_(topic_num), vnum_(word_num){}
	MapValue(uint max_local_iteration, uint topic_num, uint word_num, VectorK<double> const& alpha, MatrixKV<double> const& phi, VectorV<uint> const& word_ct, VectorK<double>& gamma)
		: knum_(topic_num), vnum_(word_num), word_ct_(&word_ct), alpha_(&alpha), phi_(&phi), gamma_(&gamma){}
};

// mapreduceの設定
struct Specification : public mapreduce::specification
{
	Specification(uint map_task_num, uint reduce_task_num){
		map_tasks = map_task_num;
		reduce_tasks = reduce_task_num;
		//output_filespec = "";
		//input_directory = "";
	}
};


namespace datasource
{
// map前にkeyとvalueを作成するクラス
class MRInputIterator
{
	std::weak_ptr<MrLDA> mrlda_;
	const uint doc_num_;
	const Specification specification_;
	static uint doc_ct_;
	static std::mutex mtx_;

public:
	MRInputIterator(std::shared_ptr<MrLDA> mrlda, Specification spec);

	bool setup_key(DocumentId &key) const
	{
		if (doc_ct_ < doc_num_){
			std::lock_guard<decltype(mtx_)> lock(mtx_);
			key = doc_ct_++;
			return true;
		}
		else{
			return false;
		}
	}

	bool get_data(DocumentId key, MapValue& value) const;

	static void reset(){ doc_ct_ = 0; }

};
}	// datasource
}	// mrlda

}
#endif