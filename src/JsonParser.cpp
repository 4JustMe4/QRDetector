#include "JsonParser.hpp"

JsonParser::JsonParser(cv::FileStorage file) { this->file = file; };

std::map<std::string, std::vector<cv::Point>> JsonParser::parsePolygon() {
  cv::FileNode root = file.root();
  root = root["_via_img_metadata"];
  std::map<std::string, std::vector<cv::Point>> standard;
  for (auto it = root.begin(); it != root.end(); it++) {
    std::string name = (std::string)(*it)["filename"];
    std::vector<cv::Point> point;
    for (int i = 0; i < 4; i++) {
      int x = (int)(*it)["regions"][0]["shape_attributes"]["all_points_x"][i];
      int y = (int)(*it)["regions"][0]["shape_attributes"]["all_points_y"][i];
      point.push_back(cv::Point(x, y));
    }
    standard[name] = point;
    point.clear();
  }
  return standard;
};