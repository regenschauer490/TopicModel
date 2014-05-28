#ifdef SIG_TOPICMODEL_MRLDA
#define SIG_TOPICMODEL_MRLDA

#include "sigtm.hpp"
#include "mapreduce_lite.hpp"

namespace sigtm
{
using MatrixVK = std::vector<std::vector<double>>;
using VectorK = std::vector<double>;
using VectorV = std::vector<double>;

class MrLDA
{
	// manage mapping procedure
	class Mapper : public mapreduce_lite::Mapper
	{
		MrLDA* outer_;
		const uint V_;
		const uint K_;
		const MatrixVK& lambda_;
		const VectorV& words_ct_;
		
	private:
		void mpa_impl(uint k_id, Document const& v_doc)
		{
			MatrixVK phi(V_, std::vector<double>(K_, 0));
			VectorK sigma(K_, 0);
			
			for(uint v=0; v<V_; ++v){
				auto total_lambda_v = sig::sum(lambda_[v]);
				for(uint k=0; k<K_; ++k){
					phi[v][k] = (lambda_[v][k] / total_lambda_v) * 
				}
			}
		}
	
	public:
		Mapper(MrLDA* outer) : outer_(outer), V_(outer->V_), K_(outer->K_), lambda_(outer->lambda_){}
		Mapper(Mapper const&) = delete;
		
		void Map(std::string const& key, std::string const& value)
		{
			
		}
	};
	
	// manage reducing procedure
	class Reducer
	{
	};

public:

private:
	const uint V_;
	const uint K_;
	
	Mapper mapper_;
	Reducer redurer_;
	
	MatrixK alpha_;			// hyper parameter of theta(topic)
	MatrixVK lambda_;		// hyper parameter of beta(word-topic)
	
public:
	MrLDA(){}
	MrLDA(MrLDA const&) = delete;
	MrLDA(MrLDA&&) = delete;
};

}
#endif