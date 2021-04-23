#pragma once
#ifndef _utils_
#define _utils_

#include <vector>
#include <string>
#include <fstream>
#include <sstream> 

std::vector<std::pair<std::string, std::vector<double>>> read_csv(std::string filename);
void print_csv(std::vector<std::pair<std::string, std::vector<double>>> dataset);
bool checkIfFIle(std::string filePath);
bool checkIfDirectory(std::string filePath);
std::string get_path(std::string dir_or_file, std::string console_string);


#endif
