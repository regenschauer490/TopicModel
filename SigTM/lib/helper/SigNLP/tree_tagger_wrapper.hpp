#ifndef SIG_TREE_TAGGER_WRAPPER_HPP
#define SIG_TREE_TAGGER_WRAPPER_HPP

#include "signlp.hpp"
#include <mutex>

namespace signlp
{
// 形態素解析器 TreeTagger ユーティリティ
class TreeTaggerWrapper
{
	const FilepassString exe_pass_;
	const FilepassString param_pass_;
	const FilepassString src_file_pass_base = SIG_TO_FPSTR("__sig_tree_tagger_tmp_src__");
	const FilepassString out_file_pass_base = SIG_TO_FPSTR("__sig_tree_tagger_tmp_out__");
	const std::string delimiter_ = " ";

	mutable std::mutex mtx_;
	mutable uint id_;

private:
	TreeTaggerWrapper(FilepassString exe_pass, FilepassString param_pass) : exe_pass_(exe_pass + SIG_TO_FPSTR(" ")), param_pass_(param_pass + SIG_TO_FPSTR(" ")), id_(0)
	{
		auto t = parseImpl(std::string("this is TreeTagger test"), SIG_TO_FPSTR(" -lemma"));
		if (t.empty()) {
			std::wcout << L"failed to execute TreeTagger" << std::endl;
			std::wcout << L"exe_pass: " << exe_pass << std::endl;
			std::wcout << L"param_pass: " << param_pass << std::endl;
			getchar();
		}
	}

	TreeTaggerWrapper(TreeTaggerWrapper const&) = delete;

	bool call(std::string const& command) const { return system(command.c_str()); }
	bool call(std::wstring const& command) const { return _wsystem(command.c_str()); }

	auto parseImpl(std::string const& src, FilepassString const& option) const->std::vector<std::string>;

public:
	static TreeTaggerWrapper& getInstance(FilepassString exe_pass, FilepassString param_pass) {
		static TreeTaggerWrapper instance(exe_pass, param_pass);	//thread safe in C++11
		return instance;
	}

	//原形に変換 (skip：原形が存在しないものは無視するか)
	auto parseGenkei(std::string const& sentence, bool skip = true) const->std::vector<std::string>;
	auto parseGenkei(std::wstring const& sentence, bool skip = true) const->std::vector<std::wstring>;
};


inline auto TreeTaggerWrapper::parseImpl(std::string const& src, FilepassString const& option) const->std::vector<std::string>
{
	{
		std::lock_guard<decltype(mtx_)> lock(mtx_);
		//if (id_ == std::numeric_limits<uint>::max()) id_ = 0;
		++id_;
	}
	auto src_file_pass = src_file_pass_base + sig::to_fpstring(id_);
	auto out_file_pass = out_file_pass_base + sig::to_fpstring(id_);

	auto div_text = sig::split(src, delimiter_);
	for (auto& e : div_text) {
		e = sig::sjis_to_utf8(e);
	}
	sig::save_line(div_text, src_file_pass);

	FilepassString command = exe_pass_ + param_pass_ + src_file_pass + SIG_TO_FPSTR(" ") + out_file_pass + option;
	uint ct = 0;
	while (true) {
		call(command);
		auto result = sig::load_line(out_file_pass);

		if (result || ct > 10) {
			while(DeleteFile(src_file_pass.c_str())) ;
			while(DeleteFile(out_file_pass.c_str())) ;

			return result ? *result : std::vector<std::string>();
		}
		++ct;
	}
}


inline auto TreeTaggerWrapper::parseGenkei(std::string const& sentence, bool skip) const->std::vector<std::string>
{
	std::vector<std::string> result;
	if (enable_warning && sentence.empty()) { std::cout << "sentense is empty" << std::endl; return result; }

	auto parse = parseImpl(sentence, SIG_TO_FPSTR(" -token -lemma -quiet"));

	for (auto& w : parse) {
		auto tmp = sig::split(w, "\t");
		if (tmp.size() == 3) {
			if(tmp[2] != "<unknown>") result.push_back(tmp[2]);
			else result.push_back(tmp[0]);
		}
	}

	return result;
}
inline auto TreeTaggerWrapper::parseGenkei(std::wstring const& sentence, bool skip) const->std::vector<std::wstring>
{
	return sig::str_to_wstr(parseGenkei(sig::wstr_to_str(sentence)));
}

}
#endif