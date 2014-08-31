/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#include "mapreduce_module.h"
#include "../model/mrlda.h"

namespace sigtm
{
namespace mrlda{

namespace datasource
{
uint MRInputIterator::doc_ct_ = 0;
std::mutex MRInputIterator::mtx_;

MRInputIterator::MRInputIterator(std::shared_ptr<MrLDA> mrlda, Specification spec) : mrlda_(mrlda), doc_num_(mrlda->D_), specification_(spec){}

bool MRInputIterator::get_data(DocumentId key, MapValue& value) const
{
	auto tmp = mrlda_.lock();
	value.knum_ = tmp->getTopicNum();
	value.vnum_ = tmp->getWordNum();
	value.word_ct_ = &tmp->doc_word_ct_[key];
	value.alpha_ = &tmp->alpha_;
	value.phi_ = &tmp->phi_;
	value.gamma_ = &tmp->gamma_[key];
		
	return true;
}

}	// datasource
}	//mrlda
}