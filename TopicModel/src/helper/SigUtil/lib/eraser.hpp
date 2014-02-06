#ifndef __SIG_UTIL_ERASER__
#define __SIG_UTIL_ERASER__

#include "sigutil.hpp"
#include <iterator>

/* �폜�E�t�B���^�֘A */

namespace sig{

	//�R���e�i�̗v�f����d���������̂��폜
	//container: �����ΏۃR���e�i
	//need_removes: �폜�����v�f��߂�l�Ŏ󂯎�邩
	//return -> �폜�v�f
	template <class C>
	C RemoveDuplicates(C& container, bool need_removes)
	{
		using T = typename container_traits<C>::value_type;
		std::unordered_map<T, bool> rmv;
		C result, removed;

		for (auto it = std::begin(container), end = std::end(container); it != end;){
			if (!rmv.count(*it)){
				container_traits<C>::add_element(result, std::move(*it));
				rmv[*it] = true;
				++it;
				continue;
			}
			else if(need_removes){
				container_traits<C>::add_element(removed, std::move(*it));
			}
			++rmv[*it];
			it = container.erase(it);
			end = container.end();
		}

		container = std::move(result);
		return std::move(removed);
	}

#if SIG_ENABLE_BOOST
#define Sig_Eraser_ParamType1 typename boost::call_traits<typename container_traits<C>::value_type>::param_type
#else
#define Sig_Eraser_ParamType1 typename std::common_type<typename container_traits<C>::value_type>::type const&
#endif

	//�R���e�i����w��v�f��1�폜
	//container: �R���e�i
	//remove: �폜�v�f
	//return -> �폜�v�f�����݂�����
	template <class C>
	bool RemoveOne(C& container, Sig_Eraser_ParamType1 remove)
	{
		for (auto it = std::begin(container), end = std::end(container); it != end;){
			if (*it == remove){
				container.erase(it);
				return true;
			}
			else ++it;
		}
		return false;
	}

	//�R���e�i����q������𖞂����v�f��1�폜
	//container: �R���e�i
	//remove_pred: �폜���ʊ֐�
	//return -> �폜�v�f�����݂�����
	template <class Pred, class C>
	bool RemoveOneIf(C& container, Pred remove_pred)
	{
		for (auto it = std::begin(container), end = std::end(container); it != end;){
			if (remove_pred(*it)){
				container.erase(it);
				return true;
			}
			else ++it;
		}
		return false;
	}

	//�R���e�i����w��v�f��S�폜
	//container: �R���e�i
	//remove: �폜�v�f
	//return -> �폜�v�f�����݂�����
	template <class C>
	bool RemoveAll(C& container, Sig_Eraser_ParamType1 remove)
	{
		uint presize = container.size();
		if (!container.empty()) Erase(container, remove);
		return presize != container.size();
	}


	//�R���e�i����q������𖞂����v�f��S�폜
	//container: �R���e�i
	//remove_pred: �폜���ʊ֐�
	//return -> �폜�v�f�����݂�����
	template <class Pred, class C>
	bool RemoveAllIf(C& container, Pred remove_pred)
	{
		uint presize = container.size();
		if (!container.empty()) EraseIf(container, remove_pred);
		return presize != container.size();
	}

}

#endif