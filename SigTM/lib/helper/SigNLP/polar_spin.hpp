#ifndef SIG_POLAR_SPIN_HPP
#define SIG_POLAR_SPIN_HPP

#include "signlp.hpp"

#if USE_SIGNLP

#include "SigUtil/lib/functional/fold.hpp"
#include "SigUtil/lib/calculation.hpp"
#include <boost/graph/adjacency_list.hpp>

namespace signlp
{

class SpinModel
{
	struct Node
	{
		std::wstring word;
		uint degree;

		bool has_label;
		double label;

		double mean_x;
		double tmp_mean_x;
	};

public:
	using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
		Node,
		boost::property<boost::edge_weight_t, double>
	>;
	using pNode = Graph::vertex_descriptor;
	using pEdge = Graph::edge_descriptor;

	static auto make_node(Graph& g, std::wstring word) ->pNode
	{
		auto v = boost::add_vertex(g);
		g[v].word = word;
		g[v].has_label = false;
		return v;
	}
	static auto make_node(Graph& g, std::wstring word, bool label) ->pNode
	{
		auto v = boost::add_vertex(g);
		g[v].word = word;
		g[v].label = label ? 1 : -1;
		g[v].has_label = true;
		return v;
	}
	static auto make_node(Graph& g, std::wstring word, double label) ->pNode
	{
		auto v = boost::add_vertex(g);
		g[v].word = word;
		g[v].label = label;
		g[v].has_label = true;
		return v;
	}

	static void make_edge(Graph& g, pNode v1, pNode v2, double weight)
	{
		auto e = add_edge(v1, v2, g);
		put(boost::edge_weight, g, e.first, weight);
	};

private:
	Graph graph_;

	const double alpha_;	// ラベル(正解)の反映度
	const double beta_;		// 逆温度

	const sig::array<int, 2> xs_;
	std::unordered_map<pNode, std::vector<pNode>> adj_;
	sig::SimpleRandom<double> rand_d_;

private:
	void init()
	{
		auto nodes = boost::vertices(graph_);

		std::for_each(
			boost::begin(nodes),
			boost::end(nodes),
			[&](pNode n){
				std::vector<pNode> tmp;
				auto adj = adjacent_vertices(n, graph_);
				for (auto it = adj.first, end = adj.second; it != end; ++it){
					tmp.push_back(*it);
				}

				graph_[n].degree = tmp.size();
				adj_.emplace(n, std::move(tmp));

				if (graph_[n].has_label){
					graph_[n].mean_x = graph_[n].label;
					graph_[n].tmp_mean_x = graph_[n].label;
				}
				else{
					double r = rand_d_();
					graph_[n].mean_x = r;
					graph_[n].tmp_mean_x = r;
				}
			}
		);

		auto edges = boost::edges(graph_);

		std::for_each(
			boost::begin(edges),
			boost::end(edges),
			[&](pEdge e){
				auto v1 = boost::source(e, graph_);
				auto v2 = boost::target(e, graph_);
				auto adjn1 = graph_[v1].degree;
				auto adjn2 = graph_[v2].degree;
	
				double decay = std::sqrt(adjn1 * adjn2);
				auto t = boost::get(boost::get(boost::edge_weight, graph_), e);
				boost::get(boost::get(boost::edge_weight, graph_), e) /= decay;				
			}
		);
	}

	void update(uint k)
	{
		const double wx_sum = sig::dotProduct(
			std::plus<double>(),
			[&](uint n){ return graph_[n].mean_x * boost::get(boost::get(boost::edge_weight, graph_), boost::edge(k, n, graph_).first); },
			0, adj_[k]
		);

		const auto exp_wx_sum = sig::map([&](int x_i){	return std::exp(beta_ * x_i * wx_sum); }, xs_);

		auto t = sig::dotProduct(std::plus<double>(), std::multiplies<double>(), 0, exp_wx_sum, xs_) / sig::sum(exp_wx_sum);

		if (!sig::is_number(t)){
			std::cout << "wxsum:" << wx_sum << std::endl;
			std::cout << "exp_wx_sum0:" << exp_wx_sum[0] << std::endl;
			std::cout << "exp_wx_sum1:" << exp_wx_sum[1] << std::endl;
			getchar();
		}

		graph_[k].tmp_mean_x = t;
	}

	void updateL(uint k)
	{
		if (adj_[k].empty()) return;
		/*const double wx_sum = sig::fold_zipWith(
			std::multiply<double>(),
			std::plus<double>(),
			0, w_[k], mean_x_
		);*/
		const double wx_sum = sig::dotProduct(
			std::plus<double>(),
			[&](uint n){ return graph_[n].mean_x * boost::get(boost::get(boost::edge_weight, graph_), boost::edge(k, n, graph_).first); },
			0, adj_[k]
		);

		const auto exp_wx_sum = sig::map([&](int x_i){	return std::exp(beta_ * x_i * wx_sum - alpha_ * std::pow(x_i - graph_[k].label, 2)); }, xs_);
		
		auto t = sig::dotProduct(std::plus<double>(), std::multiplies<double>(), 0, exp_wx_sum, xs_) / sig::sum(exp_wx_sum);

		if (!sig::is_number(t)){
			std::cout << "wxsum:" << wx_sum << std::endl;
			std::cout << "exp_wx_sum0:" << exp_wx_sum[0] << std::endl;
			std::cout << "exp_wx_sum1:" << exp_wx_sum[1] << std::endl;
			getchar();
		}

		graph_[k].tmp_mean_x = t;
	}

public:
	SpinModel(Graph const& graph, double alpha, double beta) : graph_(graph), alpha_(alpha), beta_(beta), xs_({ -1, 1 }), rand_d_(-1.0, 1.0, true)
	{
		init();
	}

	void train(uint iteration_num, std::function<void(SpinModel const*)> callback)
	{
		for (uint i = 0; i < iteration_num; ++i){
			auto nodes = boost::vertices(graph_);

			std::for_each(
				boost::begin(nodes),
				boost::end(nodes),
				[&](pNode n){ graph_[n].has_label ? updateL(n) : update(n); }
			);

			std::for_each(
				boost::begin(nodes),
				boost::end(nodes),
				[&](pNode n){ graph_[n].mean_x = graph_[n].tmp_mean_x; }
			);

			callback(this);
		}
	}

	auto getScore(std::wstring word) const->sig::Maybe<double>
	{
		auto nodes = boost::vertices(graph_);

		for(auto it = nodes.first, end = nodes.second; it != end; ++it){
			if(graph_[*it].word == word) return graph_[*it].mean_x; 
		}
		return boost::none;
	}

	auto getScore() const->std::unordered_map<std::wstring, double>
	{
		std::unordered_map<std::wstring, double> result;
		auto nodes = boost::vertices(graph_);

		for (auto it = nodes.first, end = nodes.second; it != end; ++it){
			result.emplace(graph_[*it].word, graph_[*it].mean_x);
		}
		return result;
	}

	// leave-one-out error
	double getErrorRate() const
	{
		int ct = 0;
		double sum = 0;
		auto nodes = boost::vertices(graph_);

		std::for_each(
			boost::begin(nodes),
			boost::end(nodes),
			[&](pNode n){
				if(graph_[n].has_label){
					sum += (graph_[n].label * graph_[n].mean_x < 0 ? 1 : 0);
					++ct;
				}
			}
		);

		return sum / ct;
	}

	double getMeanPolar() const
	{
		int ct = 0;
		double sum = 0;
		auto nodes = boost::vertices(graph_);

		std::for_each(
			boost::begin(nodes),
			boost::end(nodes),
			[&](pNode n){
				sum += graph_[n].mean_x;
				++ct;
			}
		);

		return sum / ct;
	}
};

}
#endif
#endif