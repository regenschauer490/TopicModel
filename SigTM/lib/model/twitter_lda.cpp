#include "twitter_lda.h"

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
	Id tkid = 0;
	for (auto const& token : tokens_){
		if (token.user_id <= uid){
			if (token.doc_id <= twid){
				++tkid;
			}
			else{
				const_cast<MatrixUD<Id>&>(T_)[uid][twid] = tkid + 1;
				tkid = 0;
				++twid;
			}
		}
		else{
			const_cast<VectorU<UserId>&>(D_)[uid] = twid + 1;
			const_cast<MatrixUD<Id>&>(T_)[uid][twid] = tkid + 1;
			twid = 0;
			tkid = 0;
			++uid;
		}
	}
	assert(U_ == uid);




	alpha_sum_ = sig::sum(alpha_);
	beta_sum_ = sig::sum(beta_);
}

inline void TwitterLDA::updateY(Token const& token, const uint t_pos)
{
	const auto sampleY = [&](const uint z, const WordId v) ->bool
	{
		auto denom = y_ct_[0] + y_ct_[1] + gamma_[0] + gamma_[1];
		auto p_y0 = ((word_ct_[v][K_] + beta_[v]) / (y_ct_[0] + beta_sum_)) * ((y_ct_[0] + gamma_[0]) / denom);
		auto p_y1 = ((word_ct_[v][z] + beta_[v]) / (topic_ct_[z] + beta_sum_)) * ((y_ct_[1] + gamma_[1]) / denom);

		double r = rand_d_() * (p_y0 + p_y1);

		return r > p_y0;	// false:general, true:topic
	};

	const auto u = token.user_id;
	const auto d = token.doc_id;
	const auto v = token.word_id;
	const auto y = y_[u][d][t_pos];
	const auto z = z_[u][d];

	if (y){
		--word_ct_[v][z];
		--topic_ct_[z];
		--y_ct_[1];
	}
	else{
		--word_ct_[v][K_];
		--y_ct_[0];
	}

	const auto new_y = sampleY(z, v);
	y_[u][d][t_pos] = new_y;
	
	if(new_y){
		++word_ct_[v][z];
		++topic_ct_[z];
		++y_ct_[1];
	}
	else{
		++word_ct_[v][K_];
		++y_ct_[0];
	}
}

inline void TwitterLDA::updateZ(const TokenIter begin, const TokenIter end)
{
	const auto sampleZ = [&](const UserId u, const DocumentId d) ->TopicId
	{
		for (TopicId k = 0; k < K_; ++k){
			auto p_z = (user_ct_[u][k] + alpha_[k]) / (D_[u] + alpha_sum_);	// todo: collapsedにできないか
			
			uint tct = 0;
			for (auto it = begin; it != end; ++it, ++tct){
				if (y_[u][d][tct]){
					const auto v = it->word_id;
					p_z *= (word_ct_[v][k] + beta_[v]) / (topic_ct_[k] + beta_sum_ + tct);
				}
			}

			tmp_p_[k] = p_z;
			if (k != 0) tmp_p_[k] += tmp_p_[k - 1];
		}

		double r = rand_d_() * tmp_p_[K_ - 1];

		for (TopicId k = 0; k < K_; ++k){
			if (r < tmp_p_[k]) return k;
		}
		return K_ - 1;
	};

	if (begin->user_id != end->user_id || begin->doc_id != end->doc_id){ std::cout << "error in updateZ" << std::endl; getchar(); }

	const auto u = begin->user_id;
	const auto d = begin->doc_id;
	const auto z = z_[u][d];

	uint tct = 0;
	for (auto it = begin; it != end; ++it, ++tct){
		if (y_[u][d][tct]){
			--word_ct_[it->word_id][z];
			--topic_ct_[z];
		}
	}
	--user_ct_[u][z];

	const auto new_z = sampleZ(u, d);
	z_[u][d] = new_z;

	tct = 0;
	for (auto it = begin; it != end; ++it, ++tct){
		if (y_[u][d][tct]){
			++word_ct_[it->word_id][new_z];
			++topic_ct_[new_z];
		}
	}
	++user_ct_[u][new_z];
}

void TwitterLDA::train(uint iteration_num, std::function<void(TwitterLDA const*)> callback)
{
	for (uint iteration = 0; iteration < iteration_num; ++iteration){
		auto token = tokens_.begin();

		for (auto tweets : T_){
			for (auto token_ct : tweets){
				updateZ(token, token + token_ct - 1);

				for (Id t = 0; t < token_ct; ++t){
					updateY(*token, t);
					++token;
				}
			}
		}

		callback(this);
	}
}

}

