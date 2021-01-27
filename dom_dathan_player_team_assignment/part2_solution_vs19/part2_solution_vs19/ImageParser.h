#pragma once
#ifndef _ImageParser_
#define _ImageParser_

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/*
Class to extract tiles from image and prepare for model prediction
*/

class ImageParser
{
private:

	// coordinates of player tile in image
	int tl_x, tl_y, br_x, br_y;

	// width and height of output image. i.e. to input into model
	const int output_width = 224; 
	const int output_height = 224;

public:
	// initialise with top left and bottom right x,y co-ordinates of tile
	ImageParser(int, int, int, int);

	// Return person tile from main image
	cv::Mat return_tile(cv::Mat img);

	// Resize image whilst maintaining aspect ratio. Add border to relevant dimension if required
	cv::Mat resizeKeepAspectRatio(const cv::Mat& input, const cv::Size& dstSize, const cv::Scalar& bgcolor);
};

#endif

