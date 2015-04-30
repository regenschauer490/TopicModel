/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#include "lda_gibbs.h"
#include "SigUtil/lib/file.hpp"

namespace sigtm
{
const auto resume_info_fname = SIG_TO_FPSTR("ldagb_info");
const auto resume_alpha_fname = SIG_TO_FPSTR("ldagb_alpha");
const auto resume_token_z_fname = SIG_TO_FPSTR("ldagb_token_z");

void LDA_Gibbs::init(bool resume)
{
	std::unordered_map<TokenId, TopicId> id_z_map;
	if (resume){
		auto const base_pass = sig::modify_dirpath_tail(input_data_->getWorkingDirectory(), true);	
		const auto load_info = sig::load_line(base_pass + resume_info_fname);

		if (isJust(load_info)){
			auto const& info = fromJust(load_info);
			total_iter_ct_ = std::stoul(info[0]);
		}

		// resume alpha
		auto load_alpha = sig::load_num<double, VectorK<double>>(base_pass + resume_alpha_fname);
		auto tmp_alpha = std::move(alpha_);
	
		if (isJust(load_alpha)){
			alpha_ = fromJust(std::move(load_alpha));
			std::cout << "resume alpha" << std::endl;
		}
		else{
			alpha_ = std::move(tmp_alpha);
			std::cout << "resume alpha error : alpha is set as default" << std::endl;
		}
	
		// resume z
		const auto load_token_z = sig::load_line(base_pass + resume_token_z_fname);
	
		if (isJust(load_token_z)){
			auto const& zs = fromJust(load_token_z);
			for(auto const& z : zs){
				const auto id_z = sig::split(z, " ");
				id_z_map.emplace(std::stoul(id_z[0]), std::stoul(id_z[1]));			
			}
			if (id_z_map.size() != tokens_.size()){ std::cout << "resume error: unmatch input data and reesume data." << std::endl; getchar(); abort(); }
			std::cout << "resume token_z" << std::endl;
		}
		else{
			std::cout << "resume token_z error : topic assigned to each token is set by random" << std::endl;
		}
	}

	alpha_sum_ = sig::sum(alpha_);
	beta_sum_ = sig::sum(beta_);

	int i = -1;	
	for(auto const& t : tokens_){
		const uint z = id_z_map.empty() ? rand_ui_() : id_z_map[t.self_id];
		++word_ct_[t.word_id][z];
		++doc_ct_[t.doc_id][z];
		++topic_ct_[z];
		z_[++i] = z;
	}
}

void LDA_Gibbs::saveResumeData() const
{
	std::cout << "save resume data... ";

	const auto base_pass = input_data_->getWorkingDirectory();

	sig::save_num(alpha_, base_pass + resume_alpha_fname, "\n");

	const auto zs = sig::map([&](Token const& t){ return std::to_string(t.self_id) + " " + std::to_string(z_[t.self_id]); }, tokens_);
	sig::save_line(zs, base_pass + resume_token_z_fname);

	sig::clear_file(base_pass + resume_info_fname);
	sig::save_line(total_iter_ct_, base_pass + resume_info_fname, sig::WriteMode::append);

	std::cout << "completed" << std::endl;
}

void LDA_Gibbs::update(Token const& t)
{
	const auto sampleTopic = [&](const DocumentId d, const WordId v)->TopicId
	{
		for (TopicId k = 0; k < K_; ++k){
			tmp_p_[k] = sampling_(this, d, v, k);
			if (k != 0) tmp_p_[k] += tmp_p_[k - 1];
		}

		double u = rand_d_() * tmp_p_[K_ - 1];

		for (TopicId k = 0; k < K_; ++k){
			if (u < tmp_p_[k]) return k;
		}
		return K_ - 1;
	};

	const auto z = z_[t.self_id];
	const auto d = t.doc_id;
	const auto v = t.word_id;

	--word_ct_[v][z];
	--doc_ct_[d][z];
	--topic_ct_[z];
    
	const auto new_z = sampleTopic(d, v);
	z_[t.self_id] = new_z;

	++word_ct_[v][new_z];
	++doc_ct_[d][new_z];
	++topic_ct_[new_z];
}


void LDA_Gibbs::train(uint num_iteration, std::function<void(LDA const*)> callback)
{
/*	auto chandle = GetStdHandle(STD_OUTPUT_HANDLE);
	CONSOLE_CURSOR_INFO info1;
	info1.dwSize = 25;
	info1.bVisible = FALSE;
	SetConsoleCursorInfo(chandle, &info1);

	CONSOLE_SCREEN_BUFFER_INFO info2;
	GetConsoleScreenBufferInfo(chandle, &info2);
	const COORD disp_pos{ 1, info2.dwCursorPosition.Y };
*/
	const auto iteration_impl = [&]{
		for (uint i = 0; i < num_iteration; ++i, ++total_iter_ct_){
			//SetConsoleCursorPosition(chandle, disp_pos);
			const std::string numstr = "iteration: " + std::to_string(total_iter_ct_ + 1);
			std::cout << numstr << std::endl;
			std::for_each(std::begin(tokens_), std::end(tokens_), std::bind(&LDA_Gibbs::update, this, std::placeholders::_1));
			callback(this);
		}
	};

	iteration_impl();
	saveResumeData();
	calcTermScore(getPhi(), term_score_);

	//SetConsoleCursorPosition(chandle, { 0, disp_pos.Y + 2 });
	//if (chandle != INVALID_HANDLE_VALUE) CloseHandle(chandle);
}

void LDA_Gibbs::save(Distribution target, FilepassString save_dir, bool detail) const
{
	save_dir = sig::modify_dirpath_tail(save_dir, true);

	switch(target){
	case Distribution::DOCUMENT :
		printTopic(
			getTheta(),
			input_data_->getInputFileNames(),
			Just(save_dir + SIG_TO_FPSTR("document_gibbs"))
		);
		break;
	case Distribution::TOPIC :
		printWord(
			getPhi(),
			std::vector<FilepassString>(),
			input_data_->words_, detail ? Nothing<uint>() : Just<uint>(20),
			Just(save_dir + SIG_TO_FPSTR("topic_gibbs"))
		);
		break;
	case Distribution::TERM_SCORE :
		printWord(
			getTermScore(),
			std::vector<FilepassString>(),
			input_data_->words_,
			detail ? Nothing<uint>() : Just<uint>(20),
			Just(save_dir + SIG_TO_FPSTR("term-score_gibbs"))
		);
		break;
	default :
		std::cout << "LDA_Gibbs::save error" << std::endl;
		getchar();
	}
}

auto LDA_Gibbs::getTheta(DocumentId d_id) const->VectorK<double>
{
	VectorK<double> theta(K_, 0);

	for(TopicId k=0; k < K_; ++k){
		theta[k] = alpha_[k] + doc_ct_[d_id][k];
	}
	sig::normalize_dist(theta);

	return theta;
}

auto LDA_Gibbs::getPhi(TopicId k_id) const->VectorV<double>
{
	VectorV<double> phi(V_, 0);

	for(WordId v=0; v < V_; ++v){
		phi[v] = beta_[v] + word_ct_[v][k_id];
	}
	sig::normalize_dist(phi);

	return std::move(phi);
}

auto LDA_Gibbs::getWordOfTopic(Distribution target, uint num_get_words, TopicId k_id) const->std::vector< std::tuple<std::wstring, double> >
{
	std::vector< std::tuple<std::wstring, double> > result;

	const std::vector<double> df =
		target == Distribution::TOPIC
		? getPhi(k_id)
		: target == Distribution::TERM_SCORE
			? getTermScore(k_id)
			: [](){
				std::cout << "LDA_Gibbs::getWordOfTopic : 不適切な'Distribution'が指定されています" << std::endl;
				std::cout << "LDA_Gibbs::getWordOfTopic : argument 'Distribution' is invalid" << std::endl;
				return std::vector<double>{};
	}();

	return calcTopWords(df, num_get_words, input_data_->words_);
}

auto LDA_Gibbs::getWordOfDocument(Distribution target, uint num_get_words, DocumentId d_id) const->std::vector< std::tuple<std::wstring, double> >
{
	std::vector< std::tuple<std::wstring, double> > result;

	const auto wscore_rank = target == Distribution::TOPIC
		? calcWordScoreOfDocument(getTheta(d_id), getPhi())
		: target == Distribution::TERM_SCORE
			? calcWordScoreOfDocument(getTheta(d_id), getTermScore())
			: [](){
				std::cout << "LDA_Gibbs::getWordOfDocument : 不適切な'Distribution'が指定されています" << std::endl;
				std::cout << "LDA_Gibbs:getWordOfDocument : argument 'Distribution' is invalid" << std::endl;
				return std::vector< std::tuple<WordId, double>>{};
	}();

	for (uint i = 0; i < num_get_words; ++i){
		result.push_back(std::make_tuple(
			*input_data_->words_.getWord(std::get<0>(wscore_rank[i])),
			std::get<1>(wscore_rank[i])
		));
	}
	return std::move(result);
}

}