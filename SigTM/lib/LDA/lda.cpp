#include <future>
#include "lda.h"

namespace sigtm
{
void LDA::initSetting()
{
	int i = -1;
	word_ct_.clear();
	doc_ct_.clear();
	topic_ct_.clear();
	z_.clear();

	for(auto const& t :tokens_){
		int assign = rand_ui_();
		++word_ct_[t->word_id][assign];
		++doc_ct_[t->doc_id][assign];
		++topic_ct_[assign];
		z_[++i] = assign;
	}
}

inline uint LDA::selectNextTopic(TokenPtr const& t)
{
	for(uint k = 0; k < T_NUM; ++k){
		p_[k] = (word_ct_[t->word_id][k] + beta_) * (doc_ct_[t->doc_id][k] + alpha_) / (topic_ct_[k] + W_NUM * beta_);
		if(k != 0) p_[k] += p_[k-1];
	}

	double u = rand_d_() * p_[T_NUM-1];

	for(uint k = 0; k < T_NUM; ++k){
		if(u < p_[k]) return k;
	}
	return T_NUM -1;
}

void LDA::resample(TokenPtr const& t)
{
	auto assign_topic = z_[t->self_id];

	--word_ct_[t->word_id][assign_topic];
	--doc_ct_[t->doc_id][assign_topic];
	--topic_ct_[assign_topic];
    
	//新しくサンプリング
	assign_topic = selectNextTopic(t);
	z_[t->self_id] = assign_topic;

	++word_ct_[t->word_id][assign_topic];
	++doc_ct_[t->doc_id][assign_topic];
	++topic_ct_[assign_topic];
  
}

void LDA::printTopicWord(Distribution dist, std::wstring const& save_pass) const
{
	std::vector< std::vector<double> > df;
	std::string fname;
	if(dist == Distribution::TOPIC){
		df = getPhi();
		fname = "lda topic";
	}
	else{
		df = getTermScoreOfTopic();
		fname = "lda term_score";
	}

	std::ostream *ofs;
	if(!save_pass.empty()){
		ofs = new std::ofstream(sig::DirpassTailModify(save_pass, true) + sig::STRtoWSTR(fname) + L".txt");
	}
	else{
		ofs = &std::cout;
	}

	for(uint k = 0; k < T_NUM; ++k){
		*ofs << fname << ": " << (k+1) << std::endl;
		std::vector< std::tuple<uint, double> > 	tmp;

		for(int w = 0; w < W_NUM; ++w){
			tmp.push_back( std::make_tuple(w, df[k][w]) );
		}
    
		std::sort(tmp.begin(), tmp.end(), [](const std::tuple<uint, double>& a, const std::tuple<uint, double>& b){ return std::get<1>(b) < std::get<1>(a); } );

		//各トピックの上位10語彙を出力
		for (uint i = 0; i < 20; ++i){
			*ofs << sig::WSTRtoSTR(*words_[std::get<0>(tmp[i])]) << ' ' << std::get<1>(tmp[i]) << std::endl;
		}
		*ofs << std::endl;
	}

	if(!save_pass.empty()) delete ofs;
}

void LDA::printDocumentTopic(std::wstring const& save_pass) const
{
	const std::vector< std::vector<double> > theta = getTheta();
	std::ostream *ofs;
	if(!save_pass.empty()){
		ofs = new std::ofstream(sig::DirpassTailModify(save_pass, true) + L"lda document' topic.txt");
	}
	else{
		ofs = &std::cout;
	}

	for(uint d = 0; d < D_NUM; ++d){
		*ofs << "document: " << (d+1) << std::endl;

		for(uint k = 0; k < T_NUM; ++k){
			if(!save_pass.empty()) *ofs << theta[d][k] << std::endl;
			else *ofs << "topic " << (k+1) << " : " << theta[d][k] << std::endl;
		}
		*ofs << std::endl;
	}

	if(!save_pass.empty()) delete ofs;
}

void LDA::printDocumentWord(std::wstring const& save_pass) const
{
	std::ostream *ofs;
	if(!save_pass.empty()){
		ofs = new std::ofstream(sig::DirpassTailModify(save_pass, true) + L"lda document' word.txt");
	}
	else{
		ofs = &std::cout;
	}

	const auto df = getWordOfDocument(25);
	for(uint d = 0; d < D_NUM; ++d){
		*ofs << "document: " << (d+1) << std::endl;

		for(uint w = 0; w < df[d].size(); ++w){
			if(!save_pass.empty()) *ofs << sig::WSTRtoSTR(std::get<0>(df[d][w])) << " : " << std::get<1>(df[d][w]) << std::endl;
			else *ofs << "word " << (w+1) << " : " << sig::WSTRtoSTR(std::get<0>(df[d][w])) << " : " << std::get<1>(df[d][w]) << std::endl;
		}
		*ofs << std::endl;
	}

	if(!save_pass.empty()) delete ofs;
}

std::vector< std::tuple<std::wstring, double> > LDA::getTopWords(std::vector<double> const& dist, uint num) const
{		
	std::vector< std::tuple<std::wstring, double> > result;
	std::vector< std::tuple<uint, double> > tmp;

	for(uint w = 0; w < W_NUM; ++w){
		tmp.push_back( std::make_tuple(w, dist[w]) );
	}
    
	std::sort(tmp.begin(), tmp.end(), [](std::tuple<uint, double> const& a, std::tuple<int, double> const& b){ return std::get<1>(b) < std::get<1>(a); } );

	for (uint i = 0; i < num; ++i){
		result.push_back(
			std::make_tuple( *words_[std::get<0>(tmp[i])], std::get<1>(tmp[i]) )
		);
	}

	return std::move(result);
}

void LDA::calcTermScore()
{
	const auto phi = getPhi();

	const auto Task_div = [&](uint const begin, uint const end){
		std::vector< std::vector<double> > ts(T_NUM);

		for(uint _w = begin, i = 0; _w < end; ++_w, ++i){
			double ip = 1;
			for(uint k2 = 0; k2 < T_NUM; ++k2){
				ip *= phi[k2][_w];
			}
			ip = pow(ip, 1.0/T_NUM);

			for(uint k = 0; k < T_NUM; ++k){
				ts[k].push_back( phi[k][_w] * log(phi[k][_w] / ip) );
				if(ts[k][i] < std::numeric_limits<double>::epsilon()) ts[k][i] = 0.0;
			}
		}
		return std::move(ts);
	};

	uint const div_size = W_NUM / THREAD_NUM;
	std::vector<std::future< std::vector<std::vector<double>> >> task;

//	std::cout << W_NUM << std::endl;
	for(uint i=0, w=0, we = div_size; i<THREAD_NUM+1; ++i, w += div_size, we += div_size){
		if(we > W_NUM) we = W_NUM;
		task.push_back( std::async(std::launch::async, Task_div, w, we) );
	}

	uint w = 0;
	for(auto& t : task){
		auto vec = t.get();
		for(uint i=0, size=vec[0].size(); i<size; ++i, ++w){
			for(uint k=0; k<T_NUM; ++k) tscore_[k][w] = vec[k][i];
		}
	}
}

void LDA::update(uint iteration_num)
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
			std::for_each(tokens_.begin(), tokens_.end(), std::bind(&LDA::_Resample, this, _1) );
		}
		_cnsl.SetCursorPosition(18, 1);
		_cos.PrintRenewLine(1u, "-- Finish --", 9, 49, aid_overwrite);
	};*/

	const auto AnimeSimple = [this,iteration_num]{
		for(uint i = 0; i < iteration_num; ++i, ++iter_ct_){
			std::string numstr = "# of iteration: " + std::to_string(iter_ct_+1);
			std::cout << numstr << std::endl;
			std::for_each(tokens_.begin(), tokens_.end(), std::bind(&LDA::resample, this, std::placeholders::_1) );
		}
	};

	AnimeSimple();
	calcTermScore();
}

double LDA::compareDistribution(CompareMethodD method, Distribution target, uint id1, uint id2) const
{
	maybe<double> val;
	std::vector< std::vector<double> > dist;

	switch(target){
	case Distribution::DOCUMENT :
		dist = getTheta();
		break;
	case Distribution::TOPIC :
		dist = getPhi();
		break;
	case Distribution::TERM_SCORE :
		dist = getTermScoreOfTopic();
		break;
	default :
		printf("\nforget: LDA::compareDistribution\n");
		getchar();
	}

	switch(method){
/*	case CompareMethod::Cos :
		val =CosineSimilarity(dist[id1], dist[id2]);
		break;*/
	case CompareMethodD::KL_DIV :
		val = KL_Divergence(dist[id1], dist[id2]);
		break;
	case CompareMethodD::JS_DIV :
		val = JS_Divergence(dist[id1], dist[id2]);
		break;
	default :
		printf("\nforget: LDA::compareDistribution\n");
		getchar();
	}

	return *val;
}

/*
std::vector< std::vector<int> > LDA::CompressTopicDimension(CompareMethodD method, double threshold) const
{
	std::vector< std::unordered_map<uint, bool> > tmp_bi(T_NUM, std::unordered_map<uint, bool>());
	std::vector< std::unordered_map<uint, bool> > tmp_mo(T_NUM, std::unordered_map<uint, bool>());

	auto dist = getTermScoreOfTopic();

	for(uint i=0; i<T_NUM; ++i){
		for(uint j=i+1; j<T_NUM; ++j){
			double cmp;
			if(method == CompareMethod::Cos) cmp = _CompareVector_Cos(dist[i], dist[j]);
			else if(method == CompareMethod::KL) cmp = _CompareDistribution_SKL(dist[i], dist[j]);

			if(cmp > threshold){
				tmp_mo[i][j] = true;
				tmp_bi[i][j] = true;
				tmp_bi[j][i] = true;
			}
			//std::cout.precision(2);
			//std::cout << cmp << " "; 
		}
		//std::cout << std::endl;
	}

	//print 類似トピック
//	for(uint i=0; i<tmp_bi.size(); ++i){
//		for(auto bit = tmp_bi[i].begin(), bend = tmp_bi[i].end(); bit != bend; ++bit) std::cout << bit->first << ",";
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;

	std::vector< std::unordered_map<uint,bool> > comb;

	auto CheckSub = [&](const std::unordered_map<uint,bool>& list)->bool{
	if(comb.empty()) return true;
		for(auto cit = comb.begin(); cit != comb.end(); ++cit){
			std::unordered_map<uint, bool> check;
			//int cf = 0;
			bool ff = true;
			if(cit->size() >= list.size()){
				for(auto ccit = cit->begin(), ccend = cit->end(); ccit != ccend; ++ccit){
					check[ccit->first] = false; 
				}
				for(auto lit = list.begin(), lend = list.end(); lit != lend; ++lit){
					if(check.count(lit->first)) ;
					else ff = false;
				}
				if(ff) return false;
			}
			else if(cit->size() < list.size()){
				for(auto lit = list.begin(), lend = list.end(); lit != lend; ++lit){
					check[lit->first] = false; 
				}
				for(auto ccit = cit->begin(), ccend = cit->end(); ccit != ccend; ++ccit){
					if(check.count(ccit->first)) ;
					else ff = false;
				}
				if(ff){
					cit = comb.erase(cit);
					if(comb.empty()) break;
					if(cit != comb.begin()) --cit;
				}
			}
		}
		return true;
	};

	std::function<void(uint, std::unordered_map<uint,bool>)> Crk = [&](uint next, std::unordered_map<uint,bool> plist){
		for(auto pit = plist.begin(), pend = plist.end(); pit != pend; ++pit){
			if(tmp_bi[pit->first].count(next)) ;
			else{
				if(CheckSub(plist))	comb.push_back(plist);
				return;
			}
		}
		plist[next] = true;
		if(tmp_mo[next].empty()){
			if(CheckSub(plist))	comb.push_back(plist);
			return;
		}
		for(auto mit = tmp_mo[next].begin(), mend = tmp_mo[next].end(); mit != mend; ++mit){
			Crk(mit->first, plist);
		}
	};
	//完全部分グラフ列挙
	for(uint i=0; i<T_NUM; ++i){
		std::unordered_map<uint,bool> list;
		Crk(i, list);
	}

	//print 完全部分グラフ
	for(uint c=0; c<comb.size(); ++c){
		for(auto ccit = comb[c].begin(), ccend = comb[c].end(); ccit != ccend; ++ccit) std::cout << ccit->first << ",";
		std::cout << std::endl;
	}
	std::cout << std::endl;

	//重複トピックを列挙
	std::vector<uint> tyofuku;
	for(uint k=0; k<T_NUM; ++k){
		uint ct = 0;
		for(auto cit = comb.begin(), cend = comb.end(); cit != cend; ++cit){
			if(cit->count(k)) ++ct;
			if(ct >= 2){
				tyofuku.push_back(k);
				break;
			}
		}
	}

	//完全グラフから重複トピックを除去
	for(auto cit = comb.begin(); cit != comb.end(); ++cit){
		for(auto tit = tyofuku.begin(), tend = tyofuku.end(); tit != tend; ){
			auto find = cit->find(*tit);
			if(find != cit->end()) cit->erase(find);
			else ++tit;
		}
	}

	std::vector< std::vector<int> > result;
	uint i = 0;
	for(auto cit = comb.begin(); cit != comb.end(); ++cit){
		if(cit->empty()) continue;
		result.push_back(std::vector<int>());
		for(auto ccit = cit->begin(), ccend = cit->end(); ccit != ccend; ++ccit){
			result[i].push_back(ccit->first);
		}
		++i;
	}

	return std::move(result);
}*/


void LDA::save(Distribution target, std::wstring const& save_pass) const
{
	switch(target){
	case Distribution::DOCUMENT :
		printDocumentTopic(save_pass);
		break;
	case Distribution::TOPIC :
		printTopicWord(target, save_pass);
		break;
	case Distribution::TERM_SCORE :
		printTopicWord(target, save_pass);
		break;
	default :
		printf("\nforget: LDA::print\n");
		getchar();
	}
}

std::vector< std::vector<double> > LDA::getTheta() const
{
	std::vector< std::vector<double> > theta;
	theta.reserve(D_NUM);

	for(uint d=0; d < D_NUM; ++d) theta.push_back(getTheta(d));
  
	return std::move(theta);
}

std::vector<double> LDA::getTheta(uint document_id) const
{
	std::vector<double> theta(T_NUM, 0);

	double sum = 0.0;
	for(uint k=0; k < T_NUM; ++k){
		theta[k] = alpha_ + doc_ct_[document_id][k];
		sum += theta[k];
	}
	//正規化
	double sinv = 1.0 / sum;
	for(uint k=0; k < T_NUM; ++k) theta[k] *= sinv;

	return std::move(theta);
}

std::vector< std::vector<double> > LDA::getPhi() const
{
	std::vector< std::vector<double> > phi;
	phi.reserve(T_NUM);
  
	for(uint k=0; k < T_NUM; ++k) phi.push_back(getPhi(k));
			
	return std::move(phi);
}

std::vector<double> LDA::getPhi(uint topic_id) const
{
	std::vector<double> phi(W_NUM, 0);

	double sum = 0.0;
	for(uint j=0; j < W_NUM; ++j){
		phi[j] = beta_ + word_ct_[j][topic_id];
		sum += phi[j];
	}
	//正規化
	double sinv = 1.0 / sum;
	for(uint j=0; j < W_NUM; ++j) phi[j] *= sinv;

	return std::move(phi);
}

std::vector< std::tuple<uint, double> > LDA::getTermScoreOfDocument(uint doc_id) const
{
	std::vector< std::tuple<uint, double> > docterm(W_NUM);
	auto theta = getTheta(doc_id);
	const auto tscore = getTermScoreOfTopic();

	std::vector<double> tmp(W_NUM, 0.0);
	uint t = 0;

	for(auto d1 = theta.begin(),  d1end = theta.end(); d1 != d1end; ++d1, ++t){
		uint w = 0;
		for(auto d2 = tscore[t].begin(), d2end = tscore[t].end(); d2 != d2end; ++d2, ++w){
			tmp[w] += ((*d1) * (*d2));
		}
	}

/*	auto th = sig::SortWithIndex(theta, false);

	for(auto d1 = th.begin(),  d1end = th.begin() + 3; d1 != d1end; ++d1, ++t){
		uint w = 0;
		for(auto d2 = tscore[std::get<0>(*d1)].begin(), d2end = tscore[std::get<0>(*d1)].end(); d2 != d2end; ++d2, ++w){
			tmp[w] += (std::get<1>(*d1) * (*d2));
		}
	}
*/
	auto stmp = sig::SortWithIndex(tmp, false);

	for (uint i = 0; i < tmp.size(); ++i){
		docterm[i] = std::make_tuple( std::get<0>(stmp[i]), std::get<1>(stmp[i]) );
	}
		
	return std::move(docterm);
}

std::vector< std::vector< std::tuple<std::wstring, double> > > LDA::getWordOfTopic(Distribution target, uint return_word_num) const
{
	std::vector< std::vector< std::tuple<std::wstring, double> > > result;

	for(uint k=0; k < T_NUM; ++k){
		result.push_back( getWordOfTopic(target, return_word_num, k) );
	}

	return std::move(result);
}

std::vector< std::tuple<std::wstring, double> > LDA::getWordOfTopic(Distribution target, uint return_word_num, uint topic_id) const
{
	std::vector< std::tuple<std::wstring, double> > result;

	std::vector<double> df;
	if(target == Distribution::TOPIC) df = getPhi(topic_id);
	else if(target == Distribution::TERM_SCORE) df = getTermScoreOfTopic(topic_id);
	else{
		std::cout << "LDA::getWordOfTopic : Distributionが無効" << std::endl;
		return result;
	}

	return getTopWords(df, return_word_num);
}

std::vector< std::vector< std::tuple<std::wstring, double> > > LDA::getWordOfDocument(uint return_word_num) const
{
	std::vector< std::vector< std::tuple<std::wstring, double> > > result;

	for(uint d=0; d<D_NUM; ++d){
		result.push_back( getWordOfDocument(return_word_num, d) );
	}

	return std::move(result);
}
	
std::vector< std::tuple<std::wstring, double> > LDA::getWordOfDocument(uint return_word_num, uint doc_id) const
{
	std::vector< std::tuple<std::wstring, double> > result;
	auto top_wscore = getTermScoreOfDocument(doc_id);

	for(uint i=0; i<return_word_num; ++i) result.push_back( std::make_tuple( *words_[std::get<0>(top_wscore[i])], std::get<1>(top_wscore[i]) ) );

	return std::move(result);
}

}	//namespace sigtm