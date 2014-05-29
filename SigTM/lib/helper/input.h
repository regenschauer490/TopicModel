/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_INPUT_CONTAINER_H
#define SIG_INPUT_CONTAINER_H

#include "../sigtm.hpp"

namespace sigtm
{
struct Token;
class InputData;

using TokenPtr = std::shared_ptr<Token const>; 
using InputDataPtr = std::shared_ptr<InputData>;


const std::function< void(std::wstring&) > df = [](std::wstring& s){};


/* ����P���\���g�[�N�� */
struct Token
{
	uint const self_id;
	uint const doc_id;
	uint const word_id;

	Token() = delete;
	Token(uint self_id, uint document_id, uint unique_word_id) : self_id(self_id), doc_id(document_id), word_id(unique_word_id){}
};
	

/* �e���f���ւ̓��̓f�[�^���쐬 */
class InputData
{
protected:
	friend class LDA;
	friend class TfIdf;

	uint doc_num_;
	std::vector<TokenPtr> tokens_;
	std::vector<C_WStrPtr> words_;
	std::unordered_map<C_WStrPtr, uint> word2id_;

private:
	InputData() = delete;
	InputData(InputData const& src) = delete;
	InputData(std::wstring const& folder_pass){ reconstruct(folder_pass); }

	bool parseLine(std::wstring const& line);
	void reconstruct(FilepassString folder_pass);


protected:
	InputData(uint doc_num) : doc_num_(doc_num){};
	void save(FilepassString folder_pass);

public:
	virtual ~InputData(){}

	// �w��̌`���̃f�[�^����ǂݍ��� or �ȑO�̒��ԏo�͂���ǂݍ���
	// folder_pass: �t�@�C�����ۑ�����Ă���f�B���N�g��
	static InputDataPtr makeInstance(FilepassString folder_pass){ return InputDataPtr(new InputData(folder_pass)); }
};

}
#endif