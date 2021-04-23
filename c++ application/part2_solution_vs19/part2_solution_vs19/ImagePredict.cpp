#include "stdafx.h"
#include <tuple>
#include "ImagePredict.h"

/*
Class to make team and person predictions on image
*/
ImagePredict::ImagePredict()
{
	// Constructor - initialise array of random colors to color person prediction
	// random seed so get the same colors each time program is run
	srand(42);

	// loop through each person in person class and each color (RBG)
	for (int i = 0; i < NUM_PERSON; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			person_color_arr[i][j] = rand() % 255; // get random number between 0 and 255 
		}
	}
}

/*
Make predictions on image
Args:
module (torch::jit::script::Module) : pytorch module loaded from a torchscript pt file
cropped_padded (cv::Mat) : player tile resized to correct size, padded if needed
Returns:
Tuple of torch tensors of predictions (team, person)
*/
std::tuple<torch::Tensor, torch::Tensor> ImagePredict::img_predict(torch::jit::script::Module module,cv::Mat cropped_padded)
{
	torch::Tensor img_tensor;

	// convert image to torch Tensor
	img_tensor = torch::from_blob(cropped_padded.data, { 1, cropped_padded.rows, cropped_padded.cols, 3 }, torch::kByte);
	// convert to CxHxW
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
	// convert to kFloat
	img_tensor = img_tensor.to(torch::kFloat);
	// Normalise
	img_tensor = img_tensor / 255; // python model ToTensor transform automatically normalised the data, hence must do the same here

	// create inputs for model
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(img_tensor);

	// make prediction
	auto outputs = module.forward(inputs).toTuple();

	// parse two outputs
	torch::Tensor output1 = outputs->elements()[0].toTensor(); // team predictions
	torch::Tensor output2 = outputs->elements()[1].toTensor(); // player predictions

	/* // one output version
	// at::Tensor output = module.forward(inputs).toTensor();
	// std::cout << output.argmax(1) << '\n';
	*/

	// std::cout << output2.argmax(1).item<int>() << std::endl;

	return { output1, output2 };
}

/*
Get colors for corresponding class of predictions
Args:
output1 (torch::Tensor) team predictions tensor
output2 (torch::Tensor) person predictions tensor
Returns:
tuple of opencv color Scalars for plotting (team color, person color)
*/
std::tuple<cv::Scalar, cv::Scalar> ImagePredict::get_prediction_colors(torch::Tensor output1, torch::Tensor output2)
{
	// initialse variables to hold team color and string prediction
	std::string team_prediction;
	cv::Scalar team_color;
	// initialise person color
	cv::Scalar person_color;

	// std::cout << output1 << std::endl;
	// get team color
	// simple switch given number of classes in this instance. 
	switch (output1.argmax(1).item<int>())
	{
	case 0:
		team_prediction = "spal_team_b";
		team_color = cv::Scalar(255, 0, 0); // blue
		break;
	case 1:
		team_prediction = "middlesbrough";
		team_color = cv::Scalar(255, 0, 255); // magenta
		break;
	case 2:
		team_prediction = "bristol";
		team_color = cv::Scalar(0, 0, 255); // red
		break;
	case 3:
		team_prediction = "nottingham_forrest";
		team_color = cv::Scalar(0, 255, 255); // yellow
		break;
	case 4:
		team_prediction = "spal_team_a";
		team_color = cv::Scalar(0, 255, 0); // green
		break;
	case 5:
		team_prediction = "wigan";
		team_color = cv::Scalar(255, 255, 0); // cyan
		break;
	}

	// std::cout << team_prediction << std::endl;

	// person color 
	// get index of maximum liklihood in prediction
	int person_index = output2.argmax(1).item<int>();

	// get person color from array
	person_color = cv::Scalar(person_color_arr[person_index][0], person_color_arr[person_index][1], person_color_arr[person_index][2]);

	return { team_color, person_color };
}