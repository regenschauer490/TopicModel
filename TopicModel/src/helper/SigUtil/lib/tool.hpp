#ifndef __SIG_UTIL_TOOL__
#define __SIG_UTIL_TOOL__

#include "sigutil.hpp"
#include <chrono>

/* �֗��c�[�� */

namespace sig{

	//���������Ɏw�肵���͈͂̈�l���z�����𔭐�������N���X
	//�f�t�H���g: ���������� -> �����Z���k�c�C�X�^�[
	template <class NumType, class Engine = std::mt19937>
	class SimpleRandom {
		Engine _engine;		//���������A���S���Y�� 
		typename std::conditional <
			std::is_integral<NumType>::value,
			std::uniform_int_distribution<int>,
			std::uniform_real_distribution<double>
		> ::type _dist;		//�m�����z

	public:
		SimpleRandom(NumType min, NumType max, bool debug) : _engine(
			[debug](){
			std::random_device rnd;
			std::vector<uint> v(10);
			if (debug) std::fill(v.begin(), v.end(), 0);
			else std::generate(v.begin(), v.end(), std::ref(rnd));

			return Engine(std::seed_seq(v.begin(), v.end()));
		}()
			),
			_dist(min, max){}

		NumType operator()(){
			return _dist(_engine);
		}
	};


	//�d���̖�����l���z�̐��������𐶐�
	template < template < class T, class = std::allocator<T>> class Container = std::vector >
	Container<int> RandomUniqueNumbers(std::size_t n, int min, int max, bool debug) {
		std::unordered_set<int> match;
		Container<int> result;
		SimpleRandom<int> Rand(0, max - min, debug);

		int r;
		for (int i = 0; i < n; ++i){
			do{
				r = min + Rand();
			} while (match.find(r) != match.end());

			match.insert(r);
			result.push_back(r);
		}

		return std::move(result);
	}

	//�^�C���E�H�b�`
	class TimeWatch{
		typedef std::chrono::system_clock::time_point Time;
		typedef decltype(std::declval<Time>() - std::declval<Time>()) Duration;

		Time st;
		std::vector<Duration> laps;
		std::vector<Duration> cache;
		bool is_run;

	private:
		void Init(){
			st = std::chrono::system_clock::now();
			laps.clear();
			cache.clear();
		}

		Duration DAccumulate(std::vector<Duration> const& ds, uint end) const{
			return std::accumulate(ds.begin(), ds.begin() + end, Duration(), [&](Duration const& sum, Duration const& v){
				return sum + v;
			});
		}

	public:
		TimeWatch(){
			Init();
			is_run = true;
		}

		//���������Ē�~
		void Reset(){
			Init();
			is_run = false;
		}

		//��~
		void Stop(){
			if (is_run)	cache.push_back(std::chrono::system_clock::now() - st);
			is_run = false;
		}
		
		//��~����
		void ReStart(){
			st = std::chrono::system_clock::now();
			is_run = true;
		}

		void Save(){
			if (is_run){
				auto now = std::chrono::system_clock::now();
				cache.push_back(now - st);
				st = std::move(now);
			}

			laps.push_back(DAccumulate(cache, cache.size()));
			cache.clear();
		}

		//�g�[�^���̎��Ԃ��擾
		//template�����Ŏ��Ԃ̒P�ʂ��w��
		template<class TimeUnit = std::chrono::milliseconds>
		long GetTotalTime(){
			return std::chrono::duration_cast<TimeUnit>(DAccumulate(laps, laps.size())).count();
		}

		//�w�肵����Ԃ܂ł̃g�[�^������(�X�v���b�g�^�C��)���擾
		//template�����Ŏ��Ԃ̒P�ʂ��w��
		template<class TimeUnit = std::chrono::milliseconds>
		auto GetSplitTime(uint index) ->typename Just<long>::type{
			return index < laps.size()
				? typename Just<long>::type(std::chrono::duration_cast<TimeUnit>(DAccumulate(laps, index+1)).count())
				: Nothing(-1);
		}

		//�w�肵����Ԃ̎���(���b�v�^�C��)���擾
		//template�����Ŏ��Ԃ̒P�ʂ��w��
		template<class TimeUnit = std::chrono::milliseconds>
		auto GetLapTime(uint index) ->typename Just<long>::type{
			return index < laps.size() ? typename Just<long>::type(std::chrono::duration_cast<TimeUnit>(laps[index]).count()) : Nothing(-1);
		}
	};

	//�q�X�g�O����
	//template <�v�f�̌^, �x��>
	template <class T, size_t BIN_NUM>
	class Histgram{
		T const _min;
		T const _max;
		double const _delta;
		std::array<uint, BIN_NUM + 2> _count;	//[0]: x < min, [BIN_NUM-1]: max <= x
		uint _num;

	private:
		void PrintBase(std::ostream& ofs) const{
			auto IsPlus = [](double v){ return v < 0 ? false : true; };

			auto IntDigit = [](double v){ return log10(v) + 1; };

			auto FirstZero = [](double v){
				uint keta = 0;
				if (Equal(v, 0)) return keta;
				while (static_cast<int>(v * std::pow(10, keta)) == 0) ++keta;
				return keta;
			};

			auto Space = [](int num){
				std::string space;
				for (int i = 0; i < num; ++i) space.append(" ");
				return std::move(space);
			};

			int const rketa = IntDigit(_max);
			int const disp_precision = typename std::conditional<std::is_integral<T>::value, std::true_type, std::false_type>::type::value
				? 0
				: IntDigit(_delta) > 1
					? 0
					: FirstZero(_delta) +1;
			int const keta = std::max(rketa, std::min((int) Precision(_delta), disp_precision) + 2);
			int const ctketa = log10(*std::max_element(_count.begin(), _count.end())) + 1;
			T const dbar = _num < 100 ? 1.0 : _num*0.01;

			/*
			std::string offset1, offset2;
			if (keta < 3) offset1.append(2 - keta, ' ');
			else offset2.append(keta - 3, ' ');*/

			ofs << "\n-- Histgram --\n";
			for (int i = 0; i < BIN_NUM + 2; ++i){
				auto low = _delta*(i - 1) + _min;
				auto high = _delta*i + _min;

				if (i == 0) ofs << std::fixed << std::setprecision(disp_precision) << "\n[-��" << Space(keta - 2) << "," << std::setw(keta + 1) << high << ")" << "�F" << std::setw(ctketa) << _count[i] << " ";
				else if (i == BIN_NUM + 1) ofs << std::fixed << std::setprecision(disp_precision) << "\n[" << std::setw(keta + 1) << low << ",+��" << Space(keta - 2) << ")" << "�F" << std::setw(ctketa) << _count[i] << " ";
				else ofs << std::fixed << std::setprecision(disp_precision) << "\n[" << std::setw(keta + 1) << low << "," << std::setw(keta + 1) << high << ")" << "�F" << std::setw(ctketa) << _count[i] << " ";

				for (int j = 1; dbar*j <= _count[i]; ++j) ofs << "|";
			}
			ofs << std::resetiosflags(std::ios_base::floatfield) << "\n\n";
		}

	public:
		//�v�f�͈̔͂��w��
		Histgram(T min, T max) : _min(min), _max(max), _delta(((double) max - min) / BIN_NUM), _num(0){
			assert(_delta > 0);
			for (auto& ct : _count) ct = 0;
		}

		//�v�f��bin�ɐU�蕪���ăJ�E���g
		void Count(T value){
			for (uint i = 0; i < BIN_NUM + 1; ++i){
				if (value < _delta*i + _min){
					++_num;
					++_count[i];
					return;
				}
			}
			++_count[BIN_NUM + 1];
		}

		template <class Container>
		void Count(Container const& values){
			for (auto const& e : values) Count(e);
		}

		//bin�O�̗v�f�����݂�����
		bool IsOverRange() const{ return _count[0] || _count[BIN_NUM + 1]; }

		//double GetAverage() const{ return std::accumulate(_count.begin(), _count.end(), 0, [](T total, T next){ return total + next; }) / static_cast<double>(_num); }

		//�p�x���擾
		auto GetCount() const -> std::array<uint, BIN_NUM>{
			std::array<uint, BIN_NUM> tmp;
			for (uint i = 0; i < BIN_NUM; ++i) tmp[i] = _count[i + 1];
			return std::move(tmp);
		}

		//bin�Ԗ�(0 �` BIN_NUM-1)�̕p�x���擾
		//return -> tuple<�p�x, �͈͍ŏ��l(�ȏ�), �͈͍ő�l(����)>
		auto GetCount(uint bin) const -> typename Just<std::tuple<uint, int, int>>::type{
			return bin < BIN_NUM
				? typename Just<std::tuple<uint, int, int>>::type(std::make_tuple(_count[bin + 1], _delta*bin + _min, _delta*(bin + 1) + _min))
				: Nothing(std::make_tuple(0, 0, 0));
		}
		
		void Print() const{ PrintBase(std::cout); }

		//�t�@�C���֏o��
		void Print(std::wstring const& file_pass) const{ PrintBase(std::ofstream(file_pass)); }
	};

}

#endif