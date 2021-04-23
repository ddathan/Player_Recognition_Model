#include "utils.h"
#include "stdafx.h"
#include <string>
#include <fstream>
#include <sstream> 
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

/*
Read csv file
Args: 
filename (string) : filename of csv
Returns:
dataset of header/value for each row
*/
std::vector<std::pair<std::string, std::vector<double>>> read_csv(std::string filename) {
	// Reads a CSV file into a vector of <string, vector<int>> pairs where
	// each pair represents <column name, column values>

	// Create a vector of <string, int vector> pairs to store the result
	std::vector<std::pair<std::string, std::vector<double>>> result;

	// Create an input filestream
	std::ifstream myFile(filename);

	// Make sure the file is open
	if (!myFile.is_open()) throw std::runtime_error("Could not open file");

	// Helper vars
	std::string line, colname;
	double val;

	// Read the column names
	if (myFile.good())
	{
		// Extract the first line in the file
		std::getline(myFile, line);

		// Create a stringstream from line
		std::stringstream ss(line);

		// Extract each column name
		while (std::getline(ss, colname, ',')) {

			// Initialize and add <colname, int vector> pairs to result
			result.push_back({ colname, std::vector<double> {} });
		}
	}

	// Read data, line by line
	while (std::getline(myFile, line))
	{
		// Create a stringstream of the current line
		std::stringstream ss(line);

		// Keep track of the current column index
		int colIdx = 0;

		// Extract each integer
		while (ss >> val) {

			// Add the current integer to the 'colIdx' column's values vector
			result.at(colIdx).second.push_back(val);

			// If the next token is a comma, ignore it and move on
			if (ss.peek() == ',') ss.ignore();

			// Increment the column index
			colIdx++;
		}
	}

	// Close file
	myFile.close();

	return result;
}

/*
* Print result of read_csv
*/
void print_csv(std::vector<std::pair<std::string, std::vector<double>>> dataset)
{
	// print data to stream
	for (int i = 0; i < dataset.at(0).second.size(); ++i)
	{
		for (int j = 0; j < dataset.size(); ++j)
		{
			std::cout << dataset.at(j).second.at(i) << "\t";
			if (j == dataset.size() - 1)
			{
				std::cout << "\n";
			}
		}
	}
}

/*
Check if given string path is of a file
*/
bool checkIfFIle(std::string filePath)
{
	try {
		// Create a Path object from given path string
		fs::path pathObj(filePath);
		// Check if path exists and is of a regular file
		if (fs::exists(pathObj) && fs::is_regular_file(pathObj))
			return true;
	}
	catch (fs::filesystem_error& e)
	{
		std::cerr << e.what() << std::endl;
	}
	return false;
}


/*
Check if given string path is of a Directory
*/
bool checkIfDirectory(std::string filePath)
{
	try {
		// Create a Path object from given path string
		fs::path pathObj(filePath);
		// Check if path exists and is of a directory file
		if (fs::exists(pathObj) && fs::is_directory(pathObj))
			return true;
	}
	catch (fs::filesystem_error& e)
	{
		std::cerr << e.what() << std::endl;
	}
	return false;
}

/*
Check if file or path is valid, wait until user inputs correct file if not
*/
std::string get_path(std::string dir_or_file, std::string console_string)
{
	bool success = false;
	std::string filename;
	std::string path;


	while (success == false)
	{
		std::cout << "Enter full path to " << console_string << std::endl;
		std::getline(std::cin, filename);

		path = filename.c_str();
		if (dir_or_file == "file")
		{
			success = checkIfFIle(path);
		}
		else
		{
			success = checkIfDirectory(path);
		}


		if (!success)
		{
			std::cout << "Cannot find " << dir_or_file << " " << filename << "\nPlease enter valid path" << std::endl;
		}

	}
	return path;
}