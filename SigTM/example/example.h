#include "make_input.hpp"


void example_lda_gibbs(InputTextType tt, FilepassString src_folder, FilepassString out_folder, sig::uint topic_num, sig::uint iteration_num, bool resume, bool make_new);
void example_lda_cvb(InputTextType tt, FilepassString src_folder, FilepassString out_folder, sig::uint topic_num, sig::uint iteration_num, bool resume, bool make_new);
void example_lda_mapreduce(InputTextType tt, FilepassString src_folder, FilepassString out_folder, sig::uint topic_num, sig::uint iteration_num,  bool resume, bool make_new);
void example_lda_twitter(FilepassString src_folder, FilepassString out_folder, sig::uint topic_num, sig::uint iteration_num, bool resume, bool make_new);
void example_ctr(FilepassString src_folder, FilepassString out_folder, sig::uint topic_num, bool run_lda, bool make_new);
