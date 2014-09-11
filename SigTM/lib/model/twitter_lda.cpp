#include "twitter_lda.h"

namespace sigtm
{
const auto resume_info_fname = SIG_STR_TO_FPSTR("twlda_info");
const auto resume_alpha_fname = SIG_STR_TO_FPSTR("twlda_alpha");
const auto resume_token_y_fname = SIG_STR_TO_FPSTR("twlda_token_y");
const auto resume_tweet_z_fname = SIG_STR_TO_FPSTR("twlda_tweet_z");

void TwitterLDA::init(bool resume)
{
	// analyze token structure
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
	if (U_ != uid){ std::cout << "error in init: input data isn't sorted." << std::endl; getchar(); abort(); }

	// resume
	std::unordered_map<TokenId, bool> id_y_map;
	std::unordered_map<std::tuple<UserId, DocumentId>, TopicId> id_z_map;
	if (resume){
		auto base_pass = sig::modify_dirpass_tail(input_data_->working_directory_, true);
	
		auto load_info = sig::read_line<std::string>(base_pass + resume_info_fname);
		if (sig::is_container_valid(load_info)){
			auto info = sig::fromJust(load_info);
			total_iter_ct_ = std::stoul(info[0]);
		}

		// resume alpha
		auto load_alpha = sig::read_num<VectorK<double>>(base_pass + resume_alpha_fname);
		auto tmp_alpha = std::move(alpha_);
	
		if (sig::is_container_valid(load_alpha)){
			alpha_ = std::move(sig::fromJust(load_alpha));
			std::cout << "resume alpha" << std::endl;
		}
		else{
			alpha_ = std::move(tmp_alpha);
			std::cout << "resume alpha error : alpha is set as default" << std::endl;
		}
	
		// resume y
		auto load_token_y = sig::read_line<std::string>(base_pass + resume_token_y_fname);
	
		if (sig::is_container_valid(load_token_y)){
			auto ys = sig::fromJust(load_token_y);
			for(auto const& y : ys){
				auto id_y = sig::split(y, " ");
				id_y_map.emplace(std::stoul(id_y[0]), std::stoul(id_y[1]));			
			}
			if (id_y_map.size() != tokens_.size()){ std::cout << "resume error: unmatch input data and reesume data." << std::endl; getchar(); abort(); }
			std::cout << "resume token_y" << std::endl;
		}
		else{
			std::cout << "resume token_y error : choice phi label to each token is set by random" << std::endl;
		}

		// resume z
		auto load_tweet_z = sig::read_line<std::string>(base_pass + resume_tweet_z_fname);

		if (sig::is_container_valid(load_tweet_z)){
			auto zs = sig::fromJust(load_tweet_z);
			for (auto const& z : zs){
				const auto u_d_z = sig::split(z, " ");
				id_z_map.emplace(std::make_tuple(std::stoul(u_d_z[0]), std::stoul(u_d_z[1])), std::stoul(u_d_z[2]));
			}
			if (id_z_map.size() != sig::sum(D_)){ std::cout << "resume error: unmatch input data and reesume data." << std::endl; getchar(); abort(); }
			std::cout << "resume tweet_z" << std::endl;
		}
		else{
			std::cout << "resume tweet_z error : topic assigned to each tweet is set by random" << std::endl;
		}
	}
	
	alpha_sum_ = sig::sum(alpha_);
	beta_sum_ = sig::sum(beta_);
	
	// init parameters
	auto token = tokens_.begin();

	for (auto tweets : T_){
		const auto u = token->user_id;

		for (auto token_ct : tweets){
			const auto d = token->doc_id;
			const auto z = id_z_map.empty() ? rand_ui_() : id_z_map[std::make_tuple(u,d)];
			z_[u][d] = z;
			++user_ct_[u][z];

			for (Id t = 0; t < token_ct; ++t){
				const auto v = token->word_id;
				const auto y = id_y_map.empty() ? rand_d_() < 0.5 : id_y_map[token->self_id];
				y_[u][d][t] = y;
				if (y){
					++word_ct_[v][z];
					++topic_ct_[z];
					++y_ct_[1];
				}
				else{
					++word_ct_[v][K_];
					++y_ct_[0];
				}
				++token;
			}
		}
	}
}

void TwitterLDA::saveResumeData() const
{
	std::cout << "save resume data... ";

	auto base_pass = input_data_->working_directory_;

	sig::save_num(alpha_, base_pass + resume_alpha_fname, "\n");
	
	VectorT<std::string> ys;
	for (auto const& t : tokens_){
		ys.push_back(std::to_string(t.self_id) + " " + std::to_string(y_[t.user_id][t.doc_id][t.word_id]));
	}
	sig::save_line(ys, base_pass + resume_token_y_fname);

	VectorD<std::string> zs;
	for (UserId u = 0; u < U_; ++u){
		for (DocumentId d = 0; d < D_[u]; ++d){
			zs.push_back(std::to_string(u) + " " + std::to_string(d) + std::to_string(z_[u][d]));
		}
	}
	sig::save_line(zs, base_pass + resume_tweet_z_fname);

	sig::clear_file(base_pass + resume_info_fname);
	sig::save_line(total_iter_ct_, base_pass + resume_info_fname, sig::WriteMode::append);

	std::cout << "completed" << std::endl;
}

inline void TwitterLDA::updateY(Token const& token, const uint t_pos)
{
	const auto sampleY = [&](const uint z, const WordId v) ->bool
	{
		auto denom = y_ct_[0] + y_ct_[1] + gamma_[0] + gamma_[1];
		auto p_y0 = ((word_ct_[v][K_] + beta_[v]) / (y_ct_[0] + beta_sum_)) * ((y_ct_[0] + gamma_[0]) / denom);
		auto p_y1 = ((word_ct_[v][z] + beta_[v]) / (topic_ct_[z] + beta_sum_)) * ((y_ct_[1] + gamma_[1]) / denom);

		double r = rand_d_() * (p_y0 + p_y1);

		return r > p_y0;	// false:background, true:topic
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

	if (begin->user_id != end->user_id || begin->doc_id != end->doc_id){ std::cout << "error in updateZ" << std::endl; getchar(); abort(); }

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

void TwitterLDA::save(Distribution target, FilepassString save_folder, bool detail) const
{
	save_folder = sig::modify_dirpass_tail(save_folder, true);

	switch (target){
	case Distribution::USER:
		printTopic(getTheta(), input_data_->doc_names_, save_folder + SIG_STR_TO_FPSTR("user_twlda"));
		break;
	case Distribution::TWEET:
		//printTopic(getTopicOfTweet(), input_data_->doc_names_, save_folder + SIG_STR_TO_FPSTR("tweet_twlda"));
		break;
	case Distribution::TOPIC:
		printWord(getPhi(), std::vector<FilepassString>(), input_data_->words_, detail ? nothing : sig::maybe<uint>(20), save_folder + SIG_STR_TO_FPSTR("topic_twlda"));
		break;
	case Distribution::TERM_SCORE:
		//printWord(getTermScore(), std::vector<FilepassString>(), input_data_->words_, detail ? nothing : sig::maybe<uint>(20), save_folder + SIG_STR_TO_FPSTR("term-score_twlda"));
		break;
	default:
		std::cout << "TwitterLDA::save error" << std::endl;
		getchar();
	}
}


auto TwitterLDA::getTheta() const->MatrixUK<double>
{
	MatrixUK<double> theta;

	for (UserId u = 0; u < U_; ++u) theta.push_back(getTheta(u));

	return theta;
}


auto TwitterLDA::getTheta(UserId u_id) const->VectorK<double>
{
	VectorK<double> theta(K_, 0);

	for (TopicId k = 0; k < K_; ++k){
		theta[k] = alpha_[k] + user_ct_[u_id][k];
	}
	sig::normalize(theta);

	return theta;
}

auto TwitterLDA::getTopicOfTweet(UserId u_id) const->MatrixDK<double>
{
	MatrixDK<double> tau;

	for (DocumentId d = 0; d < D_[u_id]; ++d) tau.push_back(getTopicOfTweet(u_id, d));

	return std::move(tau);
}

auto TwitterLDA::getTopicOfTweet(UserId u_id, DocumentId d_id) const->VectorK<double>
{
	VectorK<double> tau(K_, 0);
	const auto token = tokens_.begin();
	uint st = 0, ed = 0;

	for (uint i = 0; i<u_id; ++i) st += sig::sum(T_[i]);
	for (uint j = 0; j<d_id; ++j) st += T_[u_id][j];
	ed = st + T_[u_id][d_id];

	for (TopicId k = 0; k < K_; ++k){
		auto p_z = (user_ct_[u_id][k] + alpha_[k]) / (D_[u_id] + alpha_sum_);

		uint tct = 0;
		for (auto it = token + st, end = token + ed; it != end; ++it, ++tct){
			if (y_[u_id][d_id][tct]){
				const auto v = it->word_id;
				p_z *= (word_ct_[v][k] + beta_[v]) / (topic_ct_[k] + beta_sum_ + tct);
			}
		}
		tau[k] = p_z;
	}
	sig::normalize(tau);

	return std::move(tau);
}

auto TwitterLDA::getPhi() const->MatrixKV<double>
{
	MatrixKV<double> phi;

	for (TopicId k = 0, K = getTopicNum(); k < K; ++k) phi.push_back(getPhi(k));

	return std::move(phi);
}

auto TwitterLDA::getPhi(TopicId k_id) const->VectorV<double>
{
	VectorV<double> phi(V_, 0);

	for (WordId v = 0; v < V_; ++v){
		phi[v] = beta_[v] + word_ct_[v][k_id];
	}
	sig::normalize(phi);

	return std::move(phi);
}

auto TwitterLDA::getPhiBackground() const->VectorV<double>
{
	VectorV<double> phi(V_, 0);

	for (WordId v = 0; v < V_; ++v){
		phi[v] = beta_[v] + word_ct_[v][K_];
	}
	sig::normalize(phi);

	return std::move(phi);
}

}

