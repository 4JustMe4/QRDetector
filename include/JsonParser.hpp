#include <iostream>
#include <map>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

class JsonParser {
 private:
  cv::FileStorage file;
  JsonParser() = default;

 public:
  JsonParser(cv::FileStorage file);
  std::map<std::string, std::vector<cv::Point>> parsePolygon();
};