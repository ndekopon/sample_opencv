#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

int main(int _argc, char *_argv[])
{
	if (_argc != 3)
	{
		std::cerr << "Usage: " << _argv[0] << " <image_path> <training:0,1>" << std::endl;
		return 1;
	}

	const cv::String image_path = _argv[1];
	const cv::String training_str = _argv[2];

	bool training = training_str == "1";

	auto img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

	// check empty
	if (img.empty())
	{
		std::cerr << "Could not read the image: " << image_path << std::endl;
		return 1;
	}

	// display image info
	std::cout << "Image: " << image_path << std::endl;
	std::cout << "Width: " << img.cols << std::endl;
	std::cout << "Height: " << img.rows << std::endl;
	std::cout << "Channels: " << img.channels() << std::endl;
	std::cout << "Depth: " << img.depth() << std::endl;
	std::cout << "Type: " << img.type() << std::endl;
	std::cout << "Size: " << img.size() << std::endl;
	std::cout << "Total elements: " << img.total() << std::endl;
	std::cout << "Element size: " << img.elemSize() << std::endl;
	std::cout << "Element size1: " << img.elemSize1() << std::endl;

	// テンプレートマッチング
	cv::threshold(img, img, 254, 255, cv::THRESH_BINARY);
	cv::GaussianBlur(img, img, cv::Size(3, 3), 0);
	// cv::adaptiveThreshold(img, img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);

	if (!cv::imwrite("midimage.png", img))
	{
		std::cerr << "Could not write the image: outimage.png" << std::endl;
		return 1;
	}

	if (!training)
	{
		// 学習したファイルを読み込む
		auto loaded = cv::ml::KNearest::load("training.knn");
		if (!loaded->isTrained())
		{
			std::cerr << "Could not load the model: training.knn" << std::endl;
			return 1;
		}

		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(img, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

		cv::Mat outimage = cv::Mat::zeros(img.size(), CV_8UC3);

		for (size_t i = 0; i < contours.size(); i++)
		{
			const auto& c = contours[i];
			const auto& h = hierarchy[i];
			auto rect = cv::boundingRect(contours[i]);
			if (rect.width < 10 || 50 < rect.width) continue;
			if (rect.height < 10 || 50 < rect.height) continue;
			if (h[3] > 0) continue; // ignore holes
			auto cropped = img(rect);
			cv::Mat cropped_resized;
			cv::resize(cropped, cropped_resized, cv::Size(16, 16));
			cropped_resized = cropped_resized.reshape(1, 1);
			cropped_resized.convertTo(cropped_resized, CV_32F, 1.0 / 255.0, 0);
			cv::Mat result;
			cv::Mat neighbours;
			cv::Mat dist;
			auto f = loaded->findNearest(cropped_resized, 1, result, neighbours, dist);
			auto d = dist.at<float>(0, 0);
			auto num = cvRound(f);
			cv::String s;
			switch (num)
			{
			case '0': s = "0"; break;
			case '1': s = "1"; break;
			case '2': s = "2"; break;
			case '3': s = "3"; break;
			case '4': s = "4"; break;
			case '5': s = "5"; break;
			case '6': s = "6"; break;
			case '7': s = "7"; break;
			case '8': s = "8"; break;
			case '9': s = "9"; break;
			}
			std::cout << "Detected: " << num << std::endl;
			std::cout << "Distance: " << d << std::endl;
			cv::putText(outimage, s, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
		}


		auto result = cv::imwrite("detected.png", outimage);
		if (!result)
		{
			std::cerr << "Could not write the image: detected.png" << std::endl;
			return 1;
		}
	}
	else
	{
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(img, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

		cv::RNG rng(12345);

		cv::Mat outimage = cv::Mat::zeros(img.size(), CV_8UC3);

		cv::Mat samples;
		std::vector<float> responses;

		for (size_t i = 0; i < contours.size(); i++)
		{
			const auto& c = contours[i];
			const auto& h = hierarchy[i];
			auto rect = cv::boundingRect(contours[i]);
			if (rect.width < 8 || 50 < rect.width) continue;
			if (rect.height < 8 || 50 < rect.height) continue;
			if (h[3] > 0) continue; // ignore holes
			cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
			cv::drawContours(outimage, contours, (int)i, color, 1, cv::LINE_8, hierarchy, 0);

			auto cropped = img(rect);
			// Contoursをウィンドウ表示
			{
				// 拡大して表示(16x16 -> 64x64)
				cv::Mat cropped_resized;
				cv::resize(cropped, cropped_resized, cv::Size(16, 16));
				cv::resize(cropped_resized, cropped_resized, cv::Size(64, 64));
				cv::imshow("Contours", cropped_resized);
			}
			auto key = cv::waitKey(0);

			switch (key)
			{
			case '0':
			case '1':
			case '2':
			case '3':
			case '4':
			case '5':
			case '6':
			case '7':
			case '8':
			case '9':
				break;
			default:
				key = 0;
				break;
			}
			responses.push_back(float(key));
			cv::Mat cropped_resized;
			cv::resize(cropped, cropped_resized, cv::Size(16, 16));
			cropped_resized.reshape(1, 1).convertTo(cropped_resized, CV_32F, 1.0 / 255.0, 0);
			samples.push_back(cropped_resized);
		}
		cv::Mat responses_mat(responses.size(), 1, CV_32F, responses.data());

		std::cout << "Training samples: " << samples.size() << std::endl;
		std::cout << "Training responses: " << responses_mat.rows << std::endl;


		auto kn = cv::ml::KNearest::create();
		kn->train(samples, cv::ml::ROW_SAMPLE, responses_mat);
		kn->save("training.knn");

		auto result = cv::imwrite("outimage.png", outimage);
		if (!result)
		{
			std::cerr << "Could not write the image: outimage.png" << std::endl;
			return 1;
		}
	}

	return 0;
}
