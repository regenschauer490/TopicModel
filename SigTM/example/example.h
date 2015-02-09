#include "make_input.hpp"


void example_lda_gibbs(InputTextType tt, std::wstring src_folder, std::wstring out_folder, sig::uint topic_num, sig::uint iteration_num, bool resume, bool make_new);
void example_lda_cvb(InputTextType tt, std::wstring src_folder, std::wstring out_folder, sig::uint topic_num, sig::uint iteration_num, bool resume, bool make_new);
void example_lda_mapreduce(InputTextType tt, std::wstring src_folder, sig::uint topic_num, sig::uint iteration_num, std::wstring out_folder, bool resume, bool make_new);
void example_lda_twitter(std::wstring src_folder, std::wstring out_folder, sig::uint topic_num, sig::uint iteration_num, bool resume, bool make_new);
void example_ctr(std::wstring src_folder, std::wstring out_folder, sig::uint topic_num, bool run_lda, bool make_new);