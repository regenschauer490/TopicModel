#include "twitter_lda.h"
#include "SigUtil/lib/modify.hpp"

namespace sigtm
{

void TwitterLDA::init(bool resume)
{
	if (!input_data_->is_token_sorted_){
		std::const_pointer_cast<InputData>(input_data_)->sortToken();	// user, tweetがソートされていることが必要条件
		input_data_->save();
	}

	UserId uid = 0;
	DocumentId twid = 0;
	WordId wid = 0;
	for (auto const& token : tokens_){
		if (token.user_id <= uid){
			if (token.doc_id <= twid){
				++wid;
			}
			else{
				const_cast<MatrixUD<Id>&>(T_)[uid][twid] = wid + 1;
				wid = 0;
				++twid;
			}
		}
		else{
			const_cast<VectorU<UserId>&>(D_)[uid] = twid + 1;
			const_cast<MatrixUD<Id>&>(T_)[uid][twid] = wid + 1;
			twid = 0;
			wid = 0;
			++uid;
		}
	}
	assert(input_data_->doc_num_ == uid);
}

void TwitterLDA::updateY(Token const& t)
{

}

}

