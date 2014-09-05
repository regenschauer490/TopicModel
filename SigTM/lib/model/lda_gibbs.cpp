/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#include "lda_gibbs.h"
#include "SigUtil/lib/file.hpp"

namespace sigtm
{
const auto resume_info_fname = SIG_STR_TO_FPSTR("ldagb_info");
const auto resume_alpha_fname = SIG_STR_TO_FPSTR("ldagb_alpha");
const auto resume_token_z_fname = SIG_STR_TO_FPSTR("ldagb_token_z");

void LDA_Gibbs::init(bool resume)
{
	std::unordered_map<TokenId, TopicId> id_z_map;
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
	
		// resume z
		auto load_token_z = sig::read_line<std::string>(base_pass + resume_token_z_fname);
	
		if (sig::is_container_valid(load_token_z)){
			auto zs = sig::fromJust(load_token_z);
			for(auto const& z : zs){
				auto id_z = sig::split(z, " ");
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
		int z = id_z_map.empty() ? rand_ui_() : id_z_map[t.self_id];
		++word_ct_[t.word_id][z];
		++doc_ct_[t.doc_id][z];
		++topic_ct_[z];
		z_[++i] = z;
	}
}

void LDA_Gibbs::saveResumeData() const
{
	std::cout << "save resume data... ";

	auto base_pass = input_data_->working_directory_;

	sig::save_num(alpha_, base_pass + resume_alpha_fname, "\n");

	auto zs = sig::map([&](Token const& t){ return std::to_string(t.self_id) + " " + std::to_string(z_[t.self_id]); }, tokens_);
	sig::save_line(zs, base_pass + resume_token_z_fname);

	sig::clear_file(base_pass + resume_info_fname);
	sig::save_line(total_iter_ct_, base_pass + resume_info_fname, sig::WriteMode::append);

	std::cout << "completed" << std::endl;
}

void LDA_Gibbs::update(Token const& t)
{
	auto sampleTopic = [&](const DocumentId d, const WordId v)->TopicId
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


void LDA_Gibbs::train(uint iteration_num, std::function<void(LDA const*)> callback)
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
		for (uint i = 0; i < iteration_num; ++i, ++total_iter_ct_){
			//SetConsoleCursorPosition(chandle, disp_pos);
			std::string numstr = "iteration: " + std::to_string(total_iter_ct_ + 1);
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

void LDA_Gibbs::save(Distribution target, FilepassString save_folder, bool detail) const
{
	save_folder = sig::modify_dirpass_tail(save_folder, true);

	switch(target){
	case Distribution::DOCUMENT :
		printTopic(getTheta(), input_data_->doc_names_, save_folder + SIG_STR_TO_FPSTR("document_gibbs"));
		break;
	case Distribution::TOPIC :
		printWord(getPhi(), std::vector<FilepassString>(), input_data_->words_, sig::maybe<uint>(20), save_folder + SIG_STR_TO_FPSTR("topic_gibbs"), detail);
		break;
	case Distribution::TERM_SCORE :
		printWord(getTermScore(), std::vector<FilepassString>(), input_data_->words_, sig::maybe<uint>(20), save_folder + SIG_STR_TO_FPSTR("term-score_gibbs"), detail);
		break;
	default :
		std::cout << "LDA_Gibbs::save error" << std::endl;
		getchar();
	}
}

auto LDA_Gibbs::getTheta(DocumentId d_id) const->VectorK<double>
{
	VectorK<double> theta(K_, 0);
	double sum = 0.0;

	for(TopicId k=0; k < K_; ++k){
		theta[k] = alpha_[k] + doc_ct_[d_id][k];
		sum += theta[k];
	}
	//³‹K‰»
	double corr = 1.0 / sum;
	for(TopicId k=0; k < K_; ++k) theta[k] *= corr;

	return theta;
}

auto LDA_Gibbs::getPhi(TopicId k_id) const->VectorV<double>
{
	VectorV<double> phi(V_, 0);
	double sum = 0.0;

	for(WordId v=0; v < V_; ++v){
		phi[v] = beta_[v] + word_ct_[v][k_id];
		sum += phi[v];
	}
	//³‹K‰»
	double corr = 1.0 / sum;
	for (WordId v = 0; v < V_; ++v) phi[v] *= corr;

	return std::move(phi);
}

auto LDA_Gibbs::getWordOfTopic(Distribution target, uint return_word_num, TopicId k_id) const->std::vector< std::tuple<std::wstring, double> >
{
	std::vector< std::tuple<std::wstring, double> > result;

	std::vector<double> df;
	if (target == Distribution::TOPIC) df = getPhi(k_id);
	else if (target == Distribution::TERM_SCORE) df = getTermScore(k_id);
	else{
		std::cout << "LDA_Gibbs::getWordOfTopic : Distribution‚ª–³Œø" << std::endl;
		return result;
	}

	return getTopWords(df, return_word_num, input_data_->words_);
}

auto LDA_Gibbs::getWordOfDocument(uint return_word_num, DocumentId d_id) const->std::vector< std::tuple<std::wstring, double> >
{
	std::vector< std::tuple<std::wstring, double> > result;
	auto top_wscore = getTermScoreOfDocument(d_id);

	for(uint i=0; i<return_word_num; ++i) result.push_back( std::make_tuple( *input_data_->words_.getWord(std::get<0>(top_wscore[i])), std::get<1>(top_wscore[i]) ) );

	return std::move(result);
}

}