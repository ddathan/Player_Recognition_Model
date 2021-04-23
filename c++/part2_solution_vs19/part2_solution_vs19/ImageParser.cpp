#include "stdafx.h"
#include "ImageParser.h"

/*
Class to extract tiles from image and prepare for model prediction
*/

ImageParser::ImageParser(int a, int b, int c, int d)
// Constructor - initialise with top left and bottom right x,y co-ordinates of tile
{
	tl_x = a; // top left x position of tile
	tl_y = b; // top left y position of tile
	br_x = c; // bottom right x position of tile
	br_y = d; // bottom right y position of tile
}

/*
Return person tile from main image
Args:
img(cv::Mat): main stationary camera image
*/
cv::Mat ImageParser::return_tile(cv::Mat img)

{
	// create a region of interest around the person detection
	cv::Rect myROI(tl_x, tl_y, br_x - tl_x, br_y - tl_y);

	// create a reference to the crop around the person detection
	cv::Mat croppedRef(img, myROI);

	// copy the crop region into a new image
	cv::Mat cropped;
	croppedRef.copyTo(cropped);

	// create image of zeros the same size as required for the model: output_width x output_height
	cv::Mat cropped_padded = cv::Mat::zeros(output_width, output_height, img.type());;

	// resize the cropped image to output_width x output_height maintaining aspect ratio. Pad with zeros if required
	cropped_padded = resizeKeepAspectRatio(cropped, cropped_padded.size(), 0);

	return cropped_padded;
}


/* 
Resize image whilst maintaining aspect ratio. Add border to relevant dimension if required
Args: 
input (const cv::Mat&) : input image
dstSize (const csv::Size&) : size of output image
bgcolor (consv cv::Scalar%) : color to pad image with

*/
cv::Mat ImageParser::resizeKeepAspectRatio(const cv::Mat& input, const cv::Size& dstSize, const cv::Scalar& bgcolor)

{
	cv::Mat output;

	// get height and width
	double h1 = dstSize.width * (input.rows / (double)input.cols);
	double w2 = dstSize.height * (input.cols / (double)input.rows);

	// workout which is the limiting dimension and resize to maximise in that dimension
	if (h1 <= dstSize.height) {
		cv::resize(input, output, cv::Size(dstSize.width, h1));
	}
	else {
		cv::resize(input, output, cv::Size(w2, dstSize.height));
	}

	// initialise intergers to pad on all sides
	int top = (dstSize.height - output.rows) / 2;
	int down = (dstSize.height - output.rows + 1) / 2;
	int left = (dstSize.width - output.cols) / 2;
	int right = (dstSize.width - output.cols + 1) / 2;

	// add border
	cv::copyMakeBorder(output, output, top, down, left, right, cv::BORDER_CONSTANT, bgcolor);

	return output;
}
