#include "make_input.hpp"


void example_lda_gibbs(InputTextType tt, FilepassString src_folder, FilepassString out_folder, sig::uint num_topics, sig::uint num_iteration, bool resume, bool make_new);
void example_lda_cvb(InputTextType tt, FilepassString src_folder, FilepassString out_folder, sig::uint num_topics, sig::uint num_iteration, bool resume, bool make_new);
void example_lda_mapreduce(InputTextType tt, FilepassString src_folder, FilepassString out_folder, sig::uint num_topics, sig::uint num_iteration,  bool resume, bool make_new);
void example_lda_twitter(FilepassString src_folder, FilepassString out_folder, sig::uint num_topics, sig::uint num_iteration, bool resume, bool make_new);
