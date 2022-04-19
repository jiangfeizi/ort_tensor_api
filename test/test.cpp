#pragma execution_character_set("utf-8")  
#include "opencv2/opencv.hpp"
#include <ort_tensor_api/ort_tensor_api.hpp>
#include <iostream>
#include <string>
#include <config.h>

int main()
{
    std::string image_name = IMAGE_NAME;
	std::string model_path = MODEL_NAME;
    cv::Mat image = cv::imread(image_name, cv::IMREAD_GRAYSCALE);

    OrtTensorAPI::Resize resize;
    std::vector<cv::Mat> mats;
    mats.push_back(resize.op(image, cv::Size(28, 28), true, 0, 0, 1));

    OrtTensorAPI::Session session;
    session.init(model_path, "cpu");

    std::vector<OrtTensorAPI::InputTensor> inputs;
    inputs.push_back(OrtTensorAPI::mat_to_tensor(mats, session.m_inputs[0].m_name, { 0,0,0 }, { 255, 255, 255 }, true));

    std::vector<OrtTensorAPI::OutputTensor> outputs;
    outputs.push_back(OrtTensorAPI::OutputTensor(session.m_outputs[0].m_name));

    session.inference(inputs, outputs);

    for (int i = 0; i < 10; i++)
    {
        std::cout << "The score of " << i << " is " << ((float*)outputs[0].m_data)[i] << std::endl;
    }

	return 0;
}