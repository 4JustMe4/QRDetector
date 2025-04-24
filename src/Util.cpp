#include "Util.hpp"

void normImShow(const std::string& name, const cv::Mat& img) {
  if (img.rows > 1080) {
    cv::Mat resizedImg;
    cv::resize(img, resizedImg, cv::Size(img.cols * 1080.0 / img.rows, 1080));
    cv::imshow(name, resizedImg);
  } else {
    cv::imshow(name, img);
  }
}