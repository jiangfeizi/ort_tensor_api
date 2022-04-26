#include <ort_tensor_api/ort_tensor_api.hpp>
#include <numeric>
#include <codecvt> 
#include <config.h>

namespace OrtTensorAPI
{
	NodeInfo::NodeInfo(const std::string& name, const std::vector<int64_t>& shape) : m_name(name), m_shape(shape)
	{
	}

	NodeInfo::~NodeInfo()
	{
	}

	InputTensor::InputTensor(const std::string& name) :NodeInfo(name, {}), m_data(nullptr)
	{
	}

	InputTensor::~InputTensor()
	{
	}

	OutputTensor::OutputTensor(const std::string& name) : NodeInfo(name, {}), m_data(nullptr)
	{
	}

	OutputTensor::~OutputTensor()
	{
	}

	Session::Session()
	{
	}

	Session::~Session()
	{
		if (m_allocator && m_allocator.use_count() == 1)
		{
			for (int i = 0; i < m_session->GetInputCount(); i++)
			{
				m_allocator->Free(m_session->GetInputName(i, *m_allocator));
			}
			for (int i = 0; i < m_session->GetOutputCount(); i++)
			{
				m_allocator->Free(m_session->GetOutputName(i, *m_allocator));
			}
		}
	}

	long Session::init(const std::string& model_path, const std::string& device, GraphOptimizationLevel optimization_level)
	{
		if (m_env)
		{
			m_inputs.clear();
			m_outputs.clear();
			Session::~Session();
		}

		std::wstring wonnx_path = to_wide_string(model_path);

		m_env = std::make_shared<Ort::Env>();
		m_options = std::make_shared<Ort::SessionOptions>();
		m_options->SetGraphOptimizationLevel(optimization_level);

		if (device != "cpu")
		{
			OrtSessionOptionsAppendExecutionProvider_CUDA(*m_options, std::stoi(device));
		}

		try
		{
			m_session = std::make_shared<Ort::Session>(*m_env, wonnx_path.c_str(), *m_options);
		}
		catch (...)
		{
			return CREATE_SESSION_ERROR;
		}

		m_allocator = std::make_shared<Ort::AllocatorWithDefaultOptions>();

		for (int i = 0; i < m_session->GetInputCount(); i++)
		{
			std::string input_name = m_session->GetInputName(i, *m_allocator);
			std::vector<int64_t> input_shape = m_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
			m_inputs.push_back(NodeInfo(input_name, input_shape));
		}
		for (int i = 0; i < m_session->GetOutputCount(); i++)
		{
			std::string output_name = m_session->GetOutputName(i, *m_allocator);
			std::vector<int64_t> output_shape = m_session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
			m_outputs.push_back(NodeInfo(output_name, output_shape));
		}
		return OK;
	}

	long Session::inference(const std::vector<InputTensor>& inputs, std::vector<OutputTensor>& outputs)
	{
		Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
		std::vector<const char*> input_names;
		std::vector<Ort::Value> input_values;
		for (int i = 0; i < inputs.size(); i++)
		{
			input_names.push_back(inputs[i].m_name.c_str());
			Ort::Value input_value = Ort::Value::CreateTensor<float>(memoryInfo, inputs[i].m_data,
				std::accumulate(inputs[i].m_shape.begin(), inputs[i].m_shape.end(), 1, std::multiplies<size_t>()),
				inputs[i].m_shape.data(), inputs[i].m_shape.size());
			input_values.push_back(std::move(input_value));
		}
		std::vector<const char*> output_names;
		for (int i = 0; i < outputs.size(); i++)
		{
			output_names.push_back(outputs[i].m_name.c_str());
		}

		try
		{
			std::vector<Ort::Value> output_values = m_session->Run(Ort::RunOptions{ nullptr },
				input_names.data(), input_values.data(), inputs.size(),
				output_names.data(), outputs.size());

			for (int i = 0; i < output_values.size(); i++)
			{
				Ort::Value& output_value = output_values[i];
				outputs[i].m_shape = output_value.GetTensorTypeAndShapeInfo().GetShape();
				outputs[i].m_data = output_value.GetTensorMutableData<void>();
#ifdef DEBUG
				std::cout << "The type of output is " << output_value.GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementType() << std::endl;
#endif // DEBUG
			}
		}
		catch (...)
		{
			return INFERENCE_ERROR;
		}

		return OK;
	}

	// convert string to wstring
	std::wstring to_wide_string(const std::string& input)
	{
		std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
		return converter.from_bytes(input);
	}

	Resize::Resize() : m_input_width(0), m_input_height(0), m_target_width(0), m_target_height(0),
		m_resize_width(0), m_resize_height(0), m_wratio(0), m_hratio(0), m_dw(0), m_dh(0)
	{
	}

	Resize::~Resize()
	{
	}

	cv::Mat Resize::op(const cv::Mat& input, cv::Size dsize, bool keep_ratio, double fx, double fy, int stride, int interpolation)
	{
		m_input_width = input.cols;
		m_input_height = input.rows;
		if (dsize.width != 0 && dsize.height != 0)
		{
			cv::Mat output;
			m_target_width = int(float(dsize.width) / stride + 0.5) * stride;
			m_target_height = int(float(dsize.height) / stride + 0.5) * stride;

			if (keep_ratio)
			{
				if (float(m_target_height) / float(input.rows) >= float(m_target_width) / float(input.cols))
				{
					m_wratio = float(m_target_width) / float(input.cols);
					m_hratio = m_wratio;
					m_resize_width = m_target_width;
					m_resize_height = int(m_hratio * input.rows + 0.5);
					m_dw = 0;
					m_dh = (m_target_height - m_resize_height) / 2;
				}
				else
				{
					m_hratio = float(m_target_height) / float(input.rows);
					m_wratio = m_hratio;
					m_resize_width = int(m_wratio * input.cols + 0.5);
					m_resize_height = m_target_height;
					m_dw = (m_target_width - m_resize_width) / 2;
					m_dh = 0;
				}

				cv::Mat resize_image;
				if (input.cols != m_resize_width || input.rows != m_resize_height)
				{
					cv::resize(input, resize_image, cv::Size(m_resize_width, m_resize_height), 0, 0, interpolation);
				}
				else
				{
					resize_image = input;
				}

				output = cv::Mat::zeros(cv::Size(m_target_width, m_target_height), input.type());

				cv::Rect roi_rect = cv::Rect(cv::Point(m_dw, m_dh), cv::Point(m_dw + m_resize_width, m_dh + m_resize_height));
				resize_image.copyTo(output(roi_rect));
			}
			else
			{
				m_wratio = float(m_target_width) / float(input.cols);
				m_hratio = float(m_target_height) / float(input.rows);
				m_resize_width = m_target_width;
				m_resize_height = m_target_height;
				m_dw = 0;
				m_dh = 0;
				if (input.cols != m_target_width || input.rows != m_target_height)
				{
					cv::resize(input, output, cv::Size(m_target_width, m_target_height), 0, 0, interpolation);
				}
				else
				{
					output = input.clone();
				}
			}
			return output;
		}
		else
		{
			m_resize_width = int(input.cols * fx + 0.5);
			m_resize_height = int(input.rows * fy + 0.5);
			m_target_width = ceil(float(m_resize_width) / stride) * stride;
			m_target_height = ceil(float(m_resize_height) / stride) * stride;
			m_wratio = fx;
			m_hratio = fy;
			m_dw = (m_target_width - m_resize_width) / 2;
			m_dh = (m_target_height - m_resize_height) / 2;

			cv::Mat resize_image;
			if (input.cols != m_resize_width || input.rows != m_resize_height)
			{
				cv::resize(input, resize_image, cv::Size(m_resize_width, m_resize_height), 0, 0, interpolation);
			}
			else
			{
				resize_image = input;
			}

			cv::Mat output;
			output = cv::Mat::zeros(cv::Size(m_target_width, m_target_height), input.type());

			cv::Rect roi_rect = cv::Rect(cv::Point(m_dw, m_dh), cv::Point(m_dw + m_resize_width, m_dh + m_resize_height));
			resize_image.copyTo(output(roi_rect));

			return output;
		}
	}

	cv::Mat ret_continuous_image(cv::Mat& input)
	{
		cv::Mat continuous_image;
		if (input.isContinuous())
		{
			continuous_image = input;
		}
		else
		{
			continuous_image = input.clone();
		}
		return continuous_image;
	}

	class ToTensor : public cv::ParallelLoopBody
	{
	private:
		std::vector<cv::Mat>& m_inputs;
		InputTensor m_tensor;
		std::array<float, 3> m_mean;
		std::array<float, 3> m_std;
		bool m_flip;

		int m_height;
		int m_width;
		int m_channel;

		float m_f_table[256];
		float m_s_table[256];
		float m_t_table[256];

	public:
		ToTensor(std::vector<cv::Mat>& inputs, std::string &name, std::array<float, 3> mean, std::array<float, 3> std, bool flip) : 
			m_inputs(inputs), m_tensor(name), m_mean(mean), m_std(std), m_flip(flip)
		{
			CV_Assert(m_inputs.size() > 0);
			CV_Assert(m_inputs[0].type() == CV_8U || m_inputs[0].type() == CV_8UC3);
			m_height = m_inputs[0].rows;
			m_width = m_inputs[0].cols;
			m_channel = m_inputs[0].channels();
			for (int i = 1; i < m_inputs.size(); i++)
			{
				CV_Assert(m_inputs[i].type() == CV_8U || m_inputs[i].type() == CV_8UC3);
				CV_Assert(m_inputs[i].rows == m_height);
				CV_Assert(m_inputs[i].cols == m_width);
				CV_Assert(m_inputs[i].channels() == m_channel);
			}
			m_tensor.m_mat.create(m_inputs.size(), m_channel * m_height * m_width, CV_32F);
			m_tensor.m_data = reinterpret_cast<float*>(m_tensor.m_mat.data);
			m_tensor.m_shape = {(int64_t)m_inputs.size() ,m_inputs[0].channels() ,m_height ,m_width};

			if (m_channel == 1)
			{
				for (int i = 0; i < 256; ++i)
				{
					m_f_table[i] = (i - m_mean[0]) / m_std[0];
				}
			}
			else if (m_channel == 3)
			{
				for (int i = 0; i < 256; ++i)
				{
					m_f_table[i] = (i - m_mean[0]) / m_std[0];
					m_s_table[i] = (i - m_mean[1]) / m_std[1];
					m_t_table[i] = (i - m_mean[2]) / m_std[2];
				}
			}
		}

		virtual void operator()(const cv::Range& range) const
		{
			for (int r = range.start; r < range.end; r++)
			{
				int index = r / m_height;
				int row = r % m_height;
				float* dptr = const_cast<float*>(m_tensor.m_mat.ptr<float>(index));
				const uchar* sptr = m_inputs[index].ptr<uchar>(row);
				if (m_channel == 1)
				{
					for (int col = 0; col < m_width; col++)
					{
						dptr[row * m_width + col] = m_f_table[sptr[col]];
					}
				}
				else if (m_channel == 3)
				{
					if (m_flip)
					{
						for (int col = 0; col < m_width; col++)
						{
							dptr[0 * m_height * m_width + row * m_width + col] = m_f_table[sptr[col * m_channel + 2]];
							dptr[1 * m_height * m_width + row * m_width + col] = m_s_table[sptr[col * m_channel + 1]];
							dptr[2 * m_height * m_width + row * m_width + col] = m_t_table[sptr[col * m_channel + 0]];
						}
					}
					else
					{
						for (int col = 0; col < m_width; col++)
						{
							dptr[0 * m_height * m_width + row * m_width + col] = m_f_table[sptr[col * m_channel + 0]];
							dptr[1 * m_height * m_width + row * m_width + col] = m_s_table[sptr[col * m_channel + 1]];
							dptr[2 * m_height * m_width + row * m_width + col] = m_t_table[sptr[col * m_channel + 2]];
						}
					}
				}
			}
		}

		InputTensor ret_tensor()
		{
			return m_tensor;
		}

		cv::Range ret_range()
		{
			return cv::Range(0, m_inputs.size() * m_height);
		}
	};

	InputTensor mat_to_tensor(std::vector<cv::Mat>& inputs, std::string& name, std::array<float, 3> mean, std::array<float, 3> std, bool flip)
	{
		std::vector<cv::Mat> continuous_images;
		for (int i = 0; i < inputs.size(); i++)
		{
			continuous_images.push_back(ret_continuous_image(inputs[i]));
		}
		ToTensor to_tensor(continuous_images, name, mean, std, flip);
		parallel_for_(to_tensor.ret_range(), to_tensor);
		return to_tensor.ret_tensor();
	};
}

