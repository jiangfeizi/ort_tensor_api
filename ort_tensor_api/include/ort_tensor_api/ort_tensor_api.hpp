#pragma once

#define OK										0L									
#define CREATE_SESSION_ERROR					1L				//创建session失败	
#define INFERENCE_ERROR							2L				//推理失败							


#include <string>
#include <vector>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>


namespace OrtTensorAPI
{
	struct NodeInfo
	{
		NodeInfo(const std::string& name, const std::vector<int64_t>& shape);
		~NodeInfo();

		std::string m_name;
		std::vector<int64_t> m_shape;
	};

	struct InputTensor : NodeInfo
	{
		InputTensor(const std::string& name);
		~InputTensor();

		cv::Mat m_mat;
		float* m_data;
	};

	struct OutputTensor : NodeInfo
	{
		OutputTensor(const std::string& name);
		~OutputTensor();

		void* m_data;
	};

	class Session
	{
	public:
		Session();
		~Session();

	public:
		long init(const std::string& model_path, const std::string& device, GraphOptimizationLevel optimization_level = GraphOptimizationLevel::ORT_ENABLE_ALL);
		long inference(const std::vector<InputTensor>& inputs, std::vector<OutputTensor>& outputs);

	private:
		std::shared_ptr<Ort::Env> m_env;
		std::shared_ptr<Ort::SessionOptions> m_options;
		std::shared_ptr<Ort::Session> m_session;
		std::shared_ptr<Ort::AllocatorWithDefaultOptions> m_allocator;

	public:
		std::vector<NodeInfo> m_inputs;
		std::vector<NodeInfo> m_outputs;
	};

	std::wstring to_wide_string(const std::string& input);
	
	class Resize
	{
	public:
		Resize();
		~Resize();
		cv::Mat op(const cv::Mat& input, cv::Size dsize, bool keep_ratio = true, double fx = 0, double fy = 0,
			int stride = 32, int interpolation = cv::INTER_LINEAR);

	public:
		int m_input_width;
		int m_input_height;
		int m_target_width;
		int m_target_height;
		int m_resize_width;
		int m_resize_height;
		float m_wratio;
		float m_hratio;
		int m_dw;
		int m_dh;
	};

	InputTensor mat_to_tensor(std::vector<cv::Mat>& inputs, std::string& name, std::array<float, 3> mean, std::array<float, 3> std, bool flip);
}