#include "QrDetector.hpp"
#include "Util.hpp"

QrDetector::QrDetector(cv::Mat img) { this->img = img; };

std::vector<cv::Point> QrDetector::Detection() {
  cv::Mat dst;
  preProcess(dst);
  
  std::vector<std::vector<cv::Point>> contours;
  findSquares(dst, contours);
  std::cout << "Количество контуров: " << contours.size() << std::endl;
  cv::Mat drawing = img.clone();
  for (size_t i = 0; i < contours.size(); i++) {
    cv::polylines(drawing, contours, i, cv::Scalar(0, 255, 0), 10);
  }
  cv::imwrite("Contours.jpg", drawing);

  std::vector<cv::Point> detect_points;
  findLargestSquare(contours, detect_points);
  return detect_points;
};

void QrDetector::preProcess(cv::Mat& dst) {
  cv::Mat temp = img.clone();
  cv::cvtColor(temp, temp, cv::ColorConversionCodes::COLOR_BGR2GRAY);
  cv::imwrite("Gray.jpg", temp);
  // GaussianBlur(temp, temp, cv::Size(3, 3), 0);  // 5 5
  // cv::imwrite("Blur.jpg", temp);
  // cv::Mat adaptive;
  // cv::adaptiveThreshold(temp, adaptive, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
  //                       cv::THRESH_BINARY, 21, 2);

  // cv::imwrite("Adaptive.jpg", adaptive);

  cv::Mat canny;
  cv::Mat kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));  // 5 5
  cv::Canny(temp, canny, 200, 240, 3, false);                       // 60 200
  cv::imwrite("Canny.jpg", canny);
  cv::morphologyEx(canny, dst, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
  cv::imwrite("Morphology.jpg", dst);
}

void QrDetector::Benchmark(
    std::string folderImages,
    std::map<std::string, std::vector<cv::Point>> standard) {
  int count = 0;
  std::vector<std::string> correct;
  double qualityALL = 0;
  for (auto i : standard) {
    std::string pathImg = folderImages + i.first;
    cv::Mat img = cv::imread(pathImg);
    QrDetector tmp = QrDetector(img);
    auto detectPoints = tmp.Detection();
    double quality = tmp.intersectionOverUnion(detectPoints, i.second);
    qualityALL += quality;
    cv::polylines(img, detectPoints, true, cv::Scalar(0, 0, 255), 5);
    cv::polylines(img, i.second, true, cv::Scalar(255, 0, 0), 5);
    normImShow(pathImg + "Result", img);
    std::cout << i.first << " " << quality << "\n";
  }
  std::cout << "Average quality " << qualityALL / standard.size() << "\n";
  std::cout << "Total photos " << standard.size() << "\n";
};

double QrDetector::intersectionOverUnion(
    const std::vector<cv::Point>& predicted_points,
    const std::vector<cv::Point>& ground_truth_points) {
  if (!predicted_points.size() || !cv::isContourConvex(predicted_points)) return 0;
  std::vector<cv::Point> samePoints;
  double detectArea = cv::contourArea(predicted_points);
  double standardArea = cv::contourArea(ground_truth_points);
  double same = cv::intersectConvexConvex(predicted_points, ground_truth_points,
                                          samePoints);
  double temp = same / (detectArea + standardArea - same);
  return temp;
}