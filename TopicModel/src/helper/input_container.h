#ifndef __INPUT_CONTAINER_H__
#define __INPUT_CONTAINER_H__

/*
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <functional>
*/
#include "../sigdm.hpp"

#if USE_MECAB
#include "SigNlp/process_word.h"
#else
#include "SigNlp/signlp.hpp"
#endif

namespace sigdm{

using signlp::WordClass;

struct Token;
typedef std::shared_ptr<Token const> TokenPtr; 
class InputDataFactory;
typedef std::shared_ptr<InputDataFactory> InputDataPtr;


std::function< void(std::wstring&) > const df = [](std::wstring& s){};


/* ある単語を表すトークン */
struct Token {
	uint const self_id;
	uint const doc_id;
	uint const word_id;

	Token(uint self_id, uint document_id, uint unique_word_id) : self_id(self_id), doc_id(document_id), word_id(unique_word_id){}

private:
	Token();
};
	

/* 入力データへのフィルタ処理の設定を行うクラス */
class FilterSetting{
	friend class InputDataFactory;

	bool _base_form;
	std::unordered_set<WordClass> _selected_word_class;
	std::unordered_map< int, std::unordered_set<std::wstring> > _excepted_words;
	std::function< void(std::wstring&) > _pre_filter;
	std::function< void(std::wstring&) > _aft_filter;
		
private:
	FilterSetting();// = delete;

	//_word_class に設定された品詞であるか
	bool IsSelected_(WordClass self) const{ return _selected_word_class.count(self); }

public:
	//オブジェクトの生成
	//use_base_form：形態素解析後に単語を原型に修正するか (false:元表現, true:原形) 
	FilterSetting(bool use_base_form) : _base_form(use_base_form),  _selected_word_class(), _pre_filter(df), _aft_filter(df){};


	/* トークンリストに追加する単語に関する設定 */

	//形態素解析後、リストに追加する品詞を指定
	void AddWordClass(WordClass select){ _selected_word_class.insert(select); }

	//指定ドキュメント内で除外する単語を指定 (document_idは0から)
	void AddExceptWord(uint document_id, std::wstring const& word){ _excepted_words[document_id].insert(word); }


	/* 入力データの文字列に対して行うフィルタ処理の登録 (例：正規表現でURLを除去)  */

	//形態素解析前に行うフィルタ処理を設定
	void SetPreFilter(std::function< void(std::wstring&) > const& filter){ _pre_filter = filter; }

	//形態素解析後に行うフィルタ処理を設定
	void SetAftFilter(std::function< void(std::wstring&) > const& filter){ _aft_filter = filter; }
};


/* 入力データを内部形式へ変換するクラス */
class InputDataFactory {
	friend class LDA;
	friend class TfIdf;

	int _doc_num ;
	std::vector<TokenPtr> _tokens;
	std::vector<C_WStrPtr> _words;
	std::unordered_map<std::wstring, uint const> _word2id_map;

	FilterSetting const _filter;

private:
	InputDataFactory() = delete;
//	InputDataFactory(Document const& document, std::wstring const& save_folder, FilterPtr const& filter) : _doc_num(documents.size()), _filter(filter){ std::vector< std::vector<std::string> > input(1, documents); _MakeData(input, save_folder); }
	InputDataFactory(std::wstring const& folder_pass) : _filter(nullptr){ ReconstructData_(folder_pass); }
	InputDataFactory(InputDataFactory const& src) :_doc_num(src._doc_num),_tokens(src._tokens),_words(src._words),_filter(src._filter){};

	int ParseLine_(std::string const& line, uint& tct);
	void ReconstructData_(std::wstring const& folder_pass);

#if USE_MECAB
	InputDataFactory(Documents const& raw_texts, FilterSetting const& filter, std::wstring const& folder_pass) : _doc_num(raw_texts.size()), _filter(filter){ MakeData_(raw_texts, folder_pass); }
	
	void MakeData_(Documents const& raw_texts, std::wstring const& folder_pass);
#endif
	
public:
	/* 形態素解析前の生のドキュメント群を入力する場合 */

#if USE_MECAB
	//複数のドキュメントを入力 ( wstring raw_texts[document_id][sentence_line] ) 
	static InputDataPtr MakeInstance(Documents const& raw_texts, FilterSetting const& filter, std::wstring const& save_folder_pass){ return InputDataPtr(new InputDataFactory(raw_texts, filter, save_folder_pass)); }
	//複数のドキュメントを入力 ( string raw_texts[document_id][sentence_line] ) 
	static InputDataPtr MakeInstance(std::vector<std::vector<std::string>> const& raw_texts, FilterSetting const& filter, std::wstring const& save_folder_pass){
		Documents tmp(raw_texts.size());
		for(uint i=0; i<raw_texts.size(); ++i) std::transform(raw_texts[i].begin(), raw_texts[i].end(), std::back_inserter(tmp[i]), [](std::string const& s){ return sig::STRtoWSTR(s); });
		return InputDataPtr(new InputDataFactory(tmp, filter, save_folder_pass)); 
	}
#endif
	/* 以前の中間出力を入力とする場合 (ファイルが保存されているディレクトリを指定) */
	static InputDataPtr MakeInstance(std::wstring const& folder_pass){ return InputDataPtr(new InputDataFactory(folder_pass)); }
};

}	//namespace sigdm

#endif