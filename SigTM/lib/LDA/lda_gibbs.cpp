/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#include "lda_gibbs.h"

namespace sigtm
{
void LDA_Gibbs::initSetting()
{
	int i = -1;

	for(auto const& t :tokens_){
		int assign = rand_ui_();
		++word_ct_[t.word_id][assign];
		++doc_ct_[t.doc_id][assign];
		++topic_ct_[assign];
		z_[++i] = assign;
	}
}

inline TopicId LDA_Gibbs::selectNextTopic(Token const& t)
{
	for(TopicId k = 0; k < K_; ++k){
		tmp_p_[k] = (word_ct_[t.word_id][k] + beta_) * (doc_ct_[t.doc_id][k] + alpha_) / (topic_ct_[k] + V_ * beta_);
		if(k != 0) tmp_p_[k] += tmp_p_[k-1];
	}

	double u = rand_d_() * tmp_p_[K_-1];

	for(TopicId k = 0; k < K_; ++k){
		if(u < tmp_p_[k]) return k;
	}
	return K_ -1;
}

void LDA_Gibbs::resample(Token const& t)
{
	auto assign_topic = z_[t.self_id];

	--word_ct_[t.word_id][assign_topic];
	--doc_ct_[t.doc_id][assign_topic];
	--topic_ct_[assign_topic];
    
	//新しくサンプリング
	assign_topic = selectNextTopic(t);
	z_[t.self_id] = assign_topic;

	++word_ct_[t.word_id][assign_topic];
	++doc_ct_[t.doc_id][assign_topic];
	++topic_ct_[assign_topic];
  
}


void LDA_Gibbs::learn(uint iteration_num)
{
/*	const auto Anime = [this, iteration_num]{
		static CConsole _cnsl;
		static ConsoleOStream _cos;

		const std::string nowcomputing("-- Now LDA Computing --");
		const std::string kao_ar1 [2]  = {"三(　ε:)","_(┐「ε:)_"};
		const std::string kao_ar2 [2] = {"(:3　)三", "_(:3 」∠)_"};
		int st = 0, ed = 1, kpos = 1;
		bool sf = false, kf = true;

		for(uint i = 0; i < iteration_num; ++i, ++iter_ct_){
			_cnsl.SetCursorPosition(36 - ed, 1);
			_cos.PrintRenewLine(1u, nowcomputing.substr(st, ed), 9, 49, aid_overwrite);

			if(ed < nowcomputing.size()) ++ed;
			else sf = true;

			if(sf){
				++st;
				if(st == ed){
					sf = false;
					ed = st = 0;
				}
			}

			std::string numstr = "# of iteration: " + std::to_string(iter_ct_+1);
			_cnsl.SetCursorPosition(15, 4);
			_cos.PrintRenewLine(1u, numstr, 1, 49, aid_overwrite);

			std::string kao;
			if(kf){
				if(iter_ct_%10 == 9){
					kao = kao_ar1[1];
					kf = false;
				}
				else if(iter_ct_%10 == 8){
					kao = kao_ar1[1];
					kpos += 4;
				}
				else{
					kao = kao_ar1[0];
					kpos += 4;
				}
			}
			else{
				if(iter_ct_%10 == 9){
					kao = kao_ar2[1];
					kf = true;
				}
				else if(iter_ct_%10 == 8){
					kao = kao_ar2[1];
					kpos -= 4;
				}
				else{
					kao = kao_ar2[0];
					kpos -= 4;
				}
			}
			_cnsl.SetCursorPosition(kpos, 12);
			_cos.PrintRenewLine(1u, kao, 1, 49, aid_overwrite);

			_cnsl.SetCursorPosition(0, 0);
			std::for_each(tokens_.begin(), tokens_.end(), std::bind(&LDA_Gibbs::_Resample, this, _1) );
		}
		_cnsl.SetCursorPosition(18, 1);
		_cos.PrintRenewLine(1u, "-- Finish --", 9, 49, aid_overwrite);
	};*/

	auto chandle = GetStdHandle(STD_OUTPUT_HANDLE);
	CONSOLE_CURSOR_INFO info1;
	info1.dwSize = 25;
	info1.bVisible = FALSE;
	SetConsoleCursorInfo(chandle, &info1);

	CONSOLE_SCREEN_BUFFER_INFO info2;
	GetConsoleScreenBufferInfo(chandle, &info2);
	const COORD disp_pos{ 1, info2.dwCursorPosition.Y };

	const std::wstring perp_pass = L"perplexity.txt";
	sig::clear_file(perp_pass);
	const auto AnimeSimple = [&]{
		for (uint i = 0; i < iteration_num; ++i, ++iter_ct_){
			SetConsoleCursorPosition(chandle, disp_pos);
			std::string numstr = "iteration: " + std::to_string(iter_ct_ + 1);
			std::cout << numstr << std::endl;
			std::for_each(std::begin(tokens_), std::end(tokens_), std::bind(&LDA_Gibbs::resample, this, std::placeholders::_1));

			double perp = getPerplexity();
			auto split = sig::split(std::to_string(perp), ",");
			sig::save_line(sig::cat_str(split, ""), perp_pass, sig::WriteMode::append);
		}
	};

	AnimeSimple();
	calcTermScore(getWordDistribution(), term_score_);

	SetConsoleCursorPosition(chandle, { 0, disp_pos.Y + 2 });
	//if (chandle != INVALID_HANDLE_VALUE) CloseHandle(chandle);
}

void LDA_Gibbs::save(Distribution target, FilepassString save_folder, bool detail) const
{
	save_folder = sig::impl::modify_dirpass_tail(save_folder, true);

	switch(target){
	case Distribution::DOCUMENT :
		printTopic(getTopicDistribution(), input_data_->doc_names_, save_folder + SIG_STR_TO_FPSTR("document_gibbs"));
		break;
	case Distribution::TOPIC :
		printWord(getWordDistribution(), std::vector<FilepassString>(), input_data_->words_, sig::maybe<uint>(20), save_folder + SIG_STR_TO_FPSTR("topic_gibbs"), detail);
		break;
	case Distribution::TERM_SCORE :
		printWord(getTermScoreOfTopic(), std::vector<FilepassString>(), input_data_->words_, sig::maybe<uint>(20), save_folder + SIG_STR_TO_FPSTR("term-score_gibbs"), detail);
		break;
	default :
		printf("\nforget: LDA_Gibbs::print\n");
		getchar();
	}
}

auto LDA_Gibbs::getTopicDistribution() const->MatrixDK<double>
{
	MatrixDK<double> theta;

	for(DocumentId d=0; d < D_; ++d) theta.push_back(getTopicDistribution(d));
  
	return theta;
}

auto LDA_Gibbs::getTopicDistribution(DocumentId d_id) const->VectorK<double>
{
	VectorK<double> theta(K_, 0);
	double sum = 0.0;

	for(TopicId k=0; k < K_; ++k){
		theta[k] = alpha_ + doc_ct_[d_id][k];
		sum += theta[k];
	}
	//正規化
	double corr = 1.0 / sum;
	for(TopicId k=0; k < K_; ++k) theta[k] *= corr;

	return theta;
}

auto LDA_Gibbs::getWordDistribution() const->MatrixKV<double>
{
	MatrixKV<double> phi;
  
	for(TopicId k=0; k < K_; ++k) phi.push_back(getWordDistribution(k));
			
	return std::move(phi);
}

auto LDA_Gibbs::getWordDistribution(TopicId k_id) const->VectorV<double>
{
	VectorV<double> phi(V_, 0);
	double sum = 0.0;

	for(WordId w=0; w < V_; ++w){
		phi[w] = beta_ + word_ct_[w][k_id];
		sum += phi[w];
	}
	//正規化
	double corr = 1.0 / sum;
	for (WordId w = 0; w < V_; ++w) phi[w] *= corr;

	return std::move(phi);
}

auto LDA_Gibbs::getTermScoreOfDocument(DocumentId d_id) const->std::vector<std::tuple<WordId, double>>
{
	const auto theta = getTopicDistribution(d_id);
	const auto tscore = getTermScoreOfTopic();

	VectorV<double> tmp(V_, 0.0);
	TopicId t = 0;

	for(auto d1 = theta.begin(),  d1end = theta.end(); d1 != d1end; ++d1, ++t){
		WordId w = 0;
		for(auto d2 = tscore[t].begin(), d2end = tscore[t].end(); d2 != d2end; ++d2, ++w){
			tmp[w] += ((*d1) * (*d2));
		}
	}

	auto sorted = sig::sort_with_index(tmp); //std::tuple<std::vector<double>, std::vector<uint>>
	return sig::zipWith([](WordId w, double d){ return std::make_tuple(w, d); }, std::get<1>(sorted), std::get<0>(sorted)); //sig::zip(std::get<1>(sorted), std::get<0>(sorted));
}

auto LDA_Gibbs::getWordOfTopic(Distribution target, uint return_word_num) const->VectorK< std::vector< std::tuple<std::wstring, double> > >
{
	VectorK< std::vector< std::tuple<std::wstring, double> > > result;

	for(TopicId k=0; k < K_; ++k){
		result.push_back( getWordOfTopic(target, return_word_num, k) );
	}

	return std::move(result);
}

auto LDA_Gibbs::getWordOfTopic(Distribution target, uint return_word_num, TopicId k_id) const->std::vector< std::tuple<std::wstring, double> >
{
	std::vector< std::tuple<std::wstring, double> > result;

	std::vector<double> df;
	if(target == Distribution::TOPIC) df = getWordDistribution(k_id);
	else if(target == Distribution::TERM_SCORE) df = getTermScoreOfTopic(k_id);
	else{
		std::cout << "LDA_Gibbs::getWordOfTopic : Distributionが無効" << std::endl;
		return result;
	}

	return getTopWords(df, return_word_num, input_data_->words_);
}

auto LDA_Gibbs::getWordOfDocument(uint return_word_num) const->VectorD< std::vector< std::tuple<std::wstring, double> > >
{
	std::vector< std::vector< std::tuple<std::wstring, double> > > result;

	for(DocumentId d=0; d<D_; ++d){
		result.push_back( getWordOfDocument(return_word_num, d) );
	}

	return std::move(result);
}
	
auto LDA_Gibbs::getWordOfDocument(uint return_word_num, DocumentId d_id) const->std::vector< std::tuple<std::wstring, double> >
{
	std::vector< std::tuple<std::wstring, double> > result;
	auto top_wscore = getTermScoreOfDocument(d_id);

	for(uint i=0; i<return_word_num; ++i) result.push_back( std::make_tuple( *input_data_->words_.getWord(std::get<0>(top_wscore[i])), std::get<1>(top_wscore[i]) ) );

	return std::move(result);
}

double LDA_Gibbs::getLogLikelihood() const
{
	double log_likelihood = 0;
	const auto theta = getTopicDistribution();
	const auto beta = getWordDistribution();

	for (auto const& token : tokens_){
		const auto& theta_d = theta[token.doc_id];
		uint w = token.word_id;
		double tmp = 0;
		for (uint k = 0; k < K_; ++k){
			tmp += theta_d[k] * beta[k][w];
		}
		log_likelihood += std::log(tmp);
	}

	return log_likelihood;
}

}