#include <iostream>
#include <string>
#include <set>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/videoio.hpp>

#define TRAIN_IMAGE_WIDTH 15
#define TRAIN_IMAGE_HEIGHT 20

// 映像のうち、抜き出してチェックする箇所のリスト(x, y, width, height)
std::vector<cv::Rect> regions = {
	cv::Rect(161  + 0, 378, 20, 25),
	cv::Rect(161 + 20, 378, 20, 25),
	cv::Rect(161 + 40, 378, 20, 25), // (2) :
	cv::Rect(161 + 60, 378, 20, 25),
	cv::Rect(161 + 80, 378, 20, 25),
	cv::Rect(143, 458, 36, 48),
	cv::Rect(183  + 0, 482, 18, 23),
	cv::Rect(183 + 18, 482, 18, 23),
	cv::Rect(231  + 0, 397, 15, 20),
	cv::Rect(231 + 15, 397, 15, 20), // (9) . or :
	cv::Rect(231 + 30, 397, 15, 20),
	cv::Rect(231 + 45, 397, 15, 20),
	cv::Rect(238 + 0, 397, 15, 20),
	cv::Rect(238 + 15, 397, 15, 20), // (13) . or :
	cv::Rect(238 + 30, 397, 15, 20),
	cv::Rect(238 + 45, 397, 15, 20),
};

inline cv::Mat getdata(const cv::Mat& _image, const cv::Rect& _region)
{
	cv::Mat cropped = _image(_region);
	cv::inRange(cropped, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), cropped);
	cv::resize(cropped, cropped, cv::Size(TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT));
	cv::GaussianBlur(cropped, cropped, cv::Size(3, 3), 0);
	cropped.reshape(1, 1).convertTo(cropped, CV_32F);
	return cropped;
}

inline std::string detectdata(const cv::ml::KNearest& _kn, const cv::Mat& _image)
{
	cv::Mat result;
	cv::Mat neighbours;
	cv::Mat dist;
	auto f = _kn.findNearest(_image, 1, result, neighbours, dist);
	auto n = cvRound(f);
	cv::String s;
	switch (n)
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
	case ':': s = ":"; break;
	case '.': s = "."; break;
	}
	return s;
}

int training_video(const std::string& _path)
{
	auto videofile = cv::VideoCapture();
	videofile.open(_path, cv::CAP_FFMPEG);
	if (!videofile.isOpened())
	{
		std::cerr << "Could not open the video: " << _path << std::endl;
		return 1;
	}
	std::cerr << "Video file opened: " << _path << std::endl;

	std::set<std::vector<uchar>> allimages;

	cv::Mat frame;
	while (videofile.read(frame))
	{
		for (const auto& region : regions)
		{
			auto cropped = frame(region);

			// 白のみ抜き出す
			cv::inRange(cropped, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), cropped);

			// リサイズ
			cv::resize(cropped, cropped, cv::Size(TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT));

			// vectorに変換して格納
			allimages.emplace(cropped.begin<uchar>(), cropped.end<uchar>());
		}
	}

	// 表示＆画像の数値指定
	cv::Mat samples;
	std::vector<float> responses;
	for (auto& imagedata : allimages)
	{
		if (imagedata.size() != TRAIN_IMAGE_WIDTH * TRAIN_IMAGE_HEIGHT) continue;
		cv::Mat image(TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH, CV_8UC1, (void *)imagedata.data());
		cv::Mat output;
		cv::resize(image, output, cv::Size(128, 128), 0.0, 0.0, cv::INTER_NEAREST); // ドット表示
		cv::imshow("sample_opencv", output);
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
		case ':':
		case '.':
			break;
		default:
			key = 0;
			break;
		}
		responses.push_back(float(key));

		// ぼかしてデータを格納
		cv::GaussianBlur(image, image, cv::Size(3, 3), 0);
		cv::Mat sample;
		image.reshape(1, 1).convertTo(sample, CV_32F);
		samples.push_back(sample);
	}
	cv::Mat responses_mat(responses.size(), 1, CV_32F, responses.data());
	auto kn = cv::ml::KNearest::create();
	kn->train(samples, cv::ml::ROW_SAMPLE, responses_mat);
	kn->save("video.knn");

	return 0;
}

int detect_video(const std::string& _path)
{

	// 学習したファイルを読み込む
	auto loaded = cv::ml::KNearest::load("video.knn");
	if (!loaded->isTrained())
	{
		std::cerr << "Could not load the model: video.knn" << std::endl;
		return 1;
	}

	auto videofile = cv::VideoCapture(_path, cv::CAP_FFMPEG);
	if (!videofile.isOpened())
	{
		std::cerr << "Could not open the video: " << _path << std::endl;
		return 1;
	}

	// 判別用のリージョンを抜き出す
	const auto& region_1 = regions.at(2);
	const auto& region_2 = regions.at(9);
	const auto& region_3 = regions.at(13);

	cv::Mat frame;

	std::string prev = "";
	while (videofile.read(frame))
	{
		const auto d1 = detectdata(*loaded, getdata(frame, region_1));
		const auto d2 = detectdata(*loaded, getdata(frame, region_2));
		const auto d3 = detectdata(*loaded, getdata(frame, region_3));
		if (d1 == ":")
		{
			const auto n1 = detectdata(*loaded, getdata(frame, regions.at(0)));
			const auto n2 = detectdata(*loaded, getdata(frame, regions.at(1)));
			const auto n3 = detectdata(*loaded, getdata(frame, regions.at(3)));
			const auto n4 = detectdata(*loaded, getdata(frame, regions.at(4)));
			if (n1 == "" || n2 == "" || n3 == "" || n4 == "") continue;
			std::string s = n1 + n2 + d1 + n3 + n4;
			if (prev != s)
			{
				std::cout << s << std::endl;
				prev = s;
			}
		}
		else if (d2 == ":" || d2 == ".")
		{
			const auto n1 = detectdata(*loaded, getdata(frame, regions.at(8)));
			const auto n2 = detectdata(*loaded, getdata(frame, regions.at(10)));
			const auto n3 = detectdata(*loaded, getdata(frame, regions.at(11)));
			if (n1 == "" || n2 == "" || n3 == "") continue;
			std::string s = n1 + d2 + n2 + n3;
			if (prev != s)
			{
				std::cout << s << std::endl;
				prev = s;
			}
		}
		else if (d3 == ":" || d3 == ".")
		{
			const auto n1 = detectdata(*loaded, getdata(frame, regions.at(12)));
			const auto n2 = detectdata(*loaded, getdata(frame, regions.at(14)));
			const auto n3 = detectdata(*loaded, getdata(frame, regions.at(15)));
			if (n1 == "" || n2 == "" || n3 == "") continue;
			std::string s = n1 + d3 + n2 + n3;
			if (prev != s)
			{
				std::cout << s << std::endl;
				prev = s;
			}
		}
		else
		{
			const auto n1 = detectdata(*loaded, getdata(frame, regions.at(5)));
			const auto n2 = detectdata(*loaded, getdata(frame, regions.at(6)));
			const auto n3 = detectdata(*loaded, getdata(frame, regions.at(7)));
			if (n1 == "" || n2 == "" || n3 == "") continue;
			std::string s = n1 + "." + n2 + n3;
			if (prev != s)
			{
				std::cout << s << std::endl;
				prev = s;
			}
		}
	}
	return 0;
}

int main(int _argc, char *_argv[])
{
	if (_argc != 3)
	{
		std::cerr << "Usage: " << _argv[0] << " <avi_path> <training:0,1>" << std::endl;
		return 1;
	}

	const std::string path = _argv[1];
	const std::string training_str = _argv[2];
	bool training = training_str == "1";

	if (training)
	{
		return training_video(path);
	}
	else
	{
		return detect_video(path);
	}
	return 0;
}
