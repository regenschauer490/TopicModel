#include "example/example.h"

int main()
{	
	sig::FilepassString data_folder_pass = SIG_TO_FPSTR("../../SigTM/test_data");
	sig::FilepassString input_text_pass = data_folder_pass + SIG_TO_FPSTR("/dataset/document");
	//std::wstring input_tw_pass = data_folder_pass + SIG_TO_FPSTR("/dataset/tweet");

	setlocale(LC_ALL, "Japanese");
	
	//example_lda_gibbs(InputTextType::Tweet, input_text_pass, data_folder_pass, false, false);
	//sample4(input_tw_pass, data_folder_pass, false, false);

	return 0
}
