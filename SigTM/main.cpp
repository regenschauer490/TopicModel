#include "example/example.h"

int main()
{	
	std::wstring data_folder_pass = L"../../SigTM/test_data";
	std::wstring input_text_pass = data_folder_pass + L"/dataset/document";
	//std::wstring input_tw_pass = data_folder_pass + L"/dataset/tweet";

	setlocale(LC_ALL, "Japanese");
	
	//sample1(InputTextType::Tweet, input_text_pass, data_folder_pass, false, false);
	//sample4(input_tw_pass, data_folder_pass, false, false);
	example_ctr(data_folder_pass + L"/ctr", data_folder_pass + L"/ctr", 50, true, true);

	return 0;
}