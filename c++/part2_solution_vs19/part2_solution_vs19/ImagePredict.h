#pragma once
#ifndef _ImagePredict_
#define _ImagePredict_

#include <torch/script.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tuple>

class ImagePredict
{
private:

	// number of person classes in model
	static const int NUM_PERSON = 54;
	// array to hold person colors
	int person_color_arr[NUM_PERSON][3];
public:

	// initialise array of random colors to color person prediction
	ImagePredict();

	// Make predictions on image
	std::tuple<torch::Tensor, torch::Tensor>  img_predict(torch::jit::script::Module module,cv::Mat cropped_padded);

	// Get colors for corresponding class of predictions
	std::tuple<cv::Scalar, cv::Scalar> get_prediction_colors(torch::Tensor output1, torch::Tensor output2);
		
};

#endif
