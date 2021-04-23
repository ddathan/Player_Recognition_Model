#include "stdafx.h"
#include <iostream>
#include <stdlib.h>
#include <string>
#include <utility> 
#include <stdexcept> 
#include <filesystem>
#include <memory>
#include "utils.h"
#include "ImageParser.h"
#include "ImagePredict.h"

namespace fs = std::filesystem;

int main() {

	// Get path to model from user input
	// eg. C:\Users\ddath\Documents\source\repos\player_team_assignment\team_and_person_model.pt
	std::string model_path;
	model_path = get_path("file","model pt file");
	
	// attempt to load model. throw error if unsuccessful
	torch::jit::script::Module module;
	try {
		module = torch::jit::load(model_path);
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}
	
	// Get folder of images from user input
	// eg. C:\Users\ddath\Documents\source\repos\player_team_assignment\part2
	std::string image_folder;
	image_folder = get_path("dir", "image folder");
	// Convert string input to path
	const std::filesystem::path imdir = image_folder;
	
	// Search folder for jpg images
	std::string ext(".jpg");
	std::vector<std::string> image_names;
	std::cout << "Fetching images from folder" << std::endl;
	for (auto& p : fs::recursive_directory_iterator(imdir))
	{
		if (p.path().extension() == ext)
		{
			// std::cout << p.path().stem().string() << '\n';
			image_names.push_back(p.path().stem().string());
		}
	}
	// Calculate number of images to process and print to console
	int num_of_images = image_names.size();
	std::cout << num_of_images << " images found to process" << std::endl;

	// Create folder for processed images in same folder as images folder - i.e. parent directory of images
	fs::path imoutfolder = "part2_processed_images";
	fs::path imoutdir = imdir.parent_path() / imoutfolder;
	fs::create_directory(imoutdir); // create if it doesn't already exist
	// create separate folders for team and person predictions
	fs::path imoutfolder_team = "team_predictions";
	fs::path imoutdir_team = imdir.parent_path() / imoutfolder / imoutfolder_team;
	fs::create_directory(imoutdir_team); // create if it doesn't already exist
	fs::path imoutfolder_person = "person_predictions";
	fs::path imoutdir_person = imdir.parent_path() / imoutfolder / imoutfolder_person;
	fs::create_directory(imoutdir_person); // create if it doesn't already exist

	// Start processing the images
	std::cout << "Processing images..." << std::endl;

	// Initialise predictor class
	ImagePredict Img_pred;

	// iterate over the images
	for (int im_num = 0; im_num < num_of_images; ++im_num)
	{
		// set filenames for image jpg and corresponding csv file
		fs::path imfile(image_names[im_num] + ".jpg");
		fs::path csvfile(image_names[im_num] + ".csv");
		// set paths for image jpg file and corresponding csv file
		fs::path image_path = imdir / imfile;
		fs::path csv_path = imdir / csvfile;

		// also set path for processed images
		fs::path image_outpath_team = imoutdir_team / imfile;
		fs::path image_outpath_person = imoutdir_person / imfile;

		// check csv exists. continue to next image if it does not
		bool csv_exist = checkIfFIle(csv_path.string());
		if (!csv_exist)
		{
		std::cout << csv_path.string() << " does not exist" << std::endl;
		continue;
		}

		// Read in the corresponding csv file into dataset
		std::vector<std::pair<std::string, std::vector<double>>> dataset = read_csv(csv_path.string());

		// print values in csv. for debugging
		// print_csv(dataset);

		// read in image using opencv
		cv::Mat img_team = cv::imread(image_path.string(), cv::IMREAD_COLOR);
		// check image
		if (img_team.empty())
		{
			std::cout << "Could not read the image: " << image_path << std::endl;
			return 1;
		}

		// also read in an image for person predictions
		cv::Mat img_person = cv::imread(image_path.string(), cv::IMREAD_COLOR);

		// iterage over each person detection
		for (int i = 0; i < dataset.at(0).second.size(); ++i)
		{
			// get the coordinates
			int tl_x = (int)std::round(dataset.at(0).second.at(i));
			int tl_y = (int)std::round(dataset.at(1).second.at(i));
			int br_x = (int)std::round(dataset.at(2).second.at(i));
			int br_y = (int)std::round(dataset.at(3).second.at(i));

			// Initalise ImageParser class variable to get tile
			ImageParser Img_parse(tl_x, tl_y, br_x, br_y);

			// get the tile of the person detection
			cv::Mat tile = Img_parse.return_tile(img_team);

			/*
			// debuging code to plot and get size of resulting cropped, resized, and padded image
			imshow("Display window", cropped_padded);
			int k = cv::waitKey(0);
			std::cout << "Width : " << cropped_padded.cols << std::endl;
			std::cout << "Height: " << cropped_padded.rows << std::endl;
			*/

			// get predictions for the tile
			auto [output1, output2] = Img_pred.img_predict(module, tile);

			// get colors for each prediction
			auto [team_color, person_color] = Img_pred.get_prediction_colors(output1, output2);
			
			// create rectangle around person detection, coloured by team
			cv::rectangle(img_team,
				cv::Point(tl_x, tl_y),
				cv::Point(br_x, br_y),
				team_color,
				2,
				cv::LINE_8);

			// create rectangle around person detection, coloured by person
			cv::rectangle(img_person,
				cv::Point(tl_x, tl_y),
				cv::Point(br_x, br_y),
				person_color,
				2,
				cv::LINE_8);
	
		}
		
		/* // Plot image. used for debugging
		//imshow("Display window", img);
		////namedWindow("Display window", WINDOW_AUTOSIZE);
		//int k = cv::waitKey(0); // Wait for a keystroke in the window
		*/

		// write team images to file in image_outpath_team folder
		imwrite(image_outpath_team.string(), img_team);
		// write person images to file in image_outpath_person folder
		imwrite(image_outpath_person.string(), img_person);

	}
	// Print last statements to console
	std::cout << "Processing completed" << std::endl;
	std::cout << "Team predictions are in " << imoutdir_team.string() << std::endl;
	std::cout << "Person predictions are in " << imoutdir_person.string() << std::endl;
	system("pause");
	return 0;
}