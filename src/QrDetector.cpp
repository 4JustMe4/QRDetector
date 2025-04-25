#include "QrDetector.hpp"
#include "Util.hpp"

QrDetector::QrDetector(cv::Mat img) { this->img = img; };

std::vector<cv::Point> QrDetector::Detection() {
  {
    cv::Mat dst;
    preProcess(dst, 1);
    std::vector<std::vector<cv::Point>> contours;
    findSquares(dst, contours, true);
    std::cout << "Количество контуров для маленьких квадратов: " << contours.size() << std::endl;
    cv::Mat drawing = img.clone();
    for (size_t i = 0; i < contours.size(); i++) {
      cv::polylines(drawing, contours, i, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("ContoursSmall" + std::to_string(1) + ".jpg", drawing);
    std::vector<cv::Point> detect_points;
    if (findLargestTriple(contours, detect_points)) {
      return detect_points;
    }
  }
   for (int base : {5, 7, 10, 20}) {
    cv::Mat dst;
    preProcess(dst, base);
    std::vector<std::vector<cv::Point>> contours;
    findSquares(dst, contours);
    std::cout << "Количество контуров больших квадратов: " << contours.size() << std::endl;
    cv::Mat drawing = img.clone();
    for (size_t i = 0; i < contours.size(); i++) {
      cv::polylines(drawing, contours, i, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("ContoursBig" + std::to_string(base) + ".jpg", drawing);
    std::vector<cv::Point> detect_points;
    if (findLargestSquare(contours, detect_points)) {
      return detect_points;
    }
  }
  return std::vector<cv::Point>();
};

void QrDetector::preProcess(cv::Mat& dst, int base) {
  cv::Mat temp = img.clone();
  cv::cvtColor(temp, temp, cv::ColorConversionCodes::COLOR_BGR2GRAY);
  cv::imwrite("Gray" + std::to_string(base) + ".jpg", temp);
  // GaussianBlur(temp, temp, cv::Size(3, 3), 0);  // 5 5
  // cv::imwrite("Blur.jpg", temp);
  // cv::Mat adaptive;
  // cv::adaptiveThreshold(temp, adaptive, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
  //                       cv::THRESH_BINARY, 21, 2);

  // cv::imwrite("Adaptive.jpg", adaptive);

  cv::Mat canny;
  cv::Mat kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(base, base));  // 5 5
  cv::Canny(temp, canny, 200, 240);                       // 60 200
  cv::imwrite("Canny" + std::to_string(base) + ".jpg", canny);
  cv::morphologyEx(canny, dst, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
  cv::imwrite("Morphology" + std::to_string(base) + ".jpg", dst);
}

void QrDetector::Benchmark(
    std::string folderImages,
    std::map<std::string, std::vector<cv::Point>> standard) {
  std::vector<std::pair<std::string, std::vector<cv::Point>>> ordered(standard.begin(), standard.end());
  std::sort(ordered.begin(), ordered.end(), 
    [] (const std::pair<std::string, std::vector<cv::Point>>& a, const std::pair<std::string, std::vector<cv::Point>>& b) {
      return a.first.size() != b.first.size() ? a.first.size() < b.first.size() : a.first < b.first;
    });
  int count = 0;
  std::vector<std::string> correct;
  double qualityALL = 0;
  for (auto i : ordered) {
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