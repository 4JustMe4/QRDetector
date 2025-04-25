#include <algorithm>
#include <iostream>
#include <map>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
class QrDetector {
 private:
  cv::Mat img;
  QrDetector() = default;
  // void preProcess(cv::Mat &dst);

 public:
  QrDetector(cv::Mat img);
  void preProcess(cv::Mat& dst, int base = 5);
  std::vector<cv::Point> Detection();
  static void Benchmark(std::string folderImages,
                        std::map<std::string, std::vector<cv::Point>> standard);

  double intersectionOverUnion(
      const std::vector<cv::Point>& predicted_points,
      const std::vector<cv::Point>& ground_truth_points);
};

inline bool compareContourAreas(std::vector<cv::Point> contour1,
                                std::vector<cv::Point> contour2) {
  double i = fabs(contourArea(cv::Mat(contour1)));
  double j = fabs(contourArea(cv::Mat(contour2)));
  return (i > j);
}

inline cv::Point getCentre(const std::vector<cv::Point>& square) {
  cv::Point ans;
  for (const auto& p: square) {
    ans += p;
  }
  ans /= int(square.size());
  return ans;
}

inline double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0) {
  double dx1 = pt1.x - pt0.x;
  double dy1 = pt1.y - pt0.y;
  double dx2 = pt2.x - pt0.x;
  double dy2 = pt2.y - pt0.y;
  return (dx1 * dx2 + dy1 * dy2) /
         sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}


inline double distance(cv::Point p1, cv::Point p2) {
  return std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2));
}

inline void findSquares(const cv::Mat& src,
                        std::vector<std::vector<cv::Point>>& squares,
                        bool isSmall = false) {
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(src, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

  for (size_t i = 0; i < contours.size(); i++) {
    std::vector<cv::Point> hull;
    std::vector<cv::Point> approx;
    cv::convexHull(contours[i], hull);
    cv::approxPolyDP(cv::Mat(hull), approx,
                     cv::arcLength(cv::Mat(hull), true) * 0.02, true);

    long long int size = src.rows * src.cols;
    if (approx.size() == 4 && cv::isContourConvex(cv::Mat(approx)) &&
        cv::contourArea(approx) > 600) {
      double side1 = cv::norm(approx[0] - approx[1]);
      double side2 = cv::norm(approx[1] - approx[2]);
      double side3 = cv::norm(approx[2] - approx[3]);
      double side4 = cv::norm(approx[3] - approx[0]);

      double epsilon = 0.2;  // 0.12

      if (std::fabs(side1 - side2) < epsilon * std::max(side1, side2) &&
          std::fabs(side2 - side3) < epsilon * std::max(side2, side3) &&
          std::fabs(side3 - side4) < epsilon * std::max(side3, side4)) {
        double maxCosine = 0;
        for (int j = 2; j < 5; j++) {
          double cosine =
              std::fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
          maxCosine = std::max(maxCosine, cosine);
        }

        if (maxCosine < 0.1 || !isSmall) squares.push_back(approx);
      }
    }
  }
}

inline bool findLargestTriple(
    const std::vector<std::vector<cv::Point>>& squares,
    std::vector<cv::Point>& biggest_square) {
  if (!squares.size()) {
    std::cout << "No squares detect" << std::endl;
    return false;
  }
  std::vector<cv::Point> ans;
  std::vector<double> epsilons = {0.1, 0.2, 0.3};
  for (auto eps: epsilons) {
    for (size_t i = 0; i < squares.size(); i++) {
      for (size_t j = 0; j < squares.size(); j++) {
        if (i == j) continue;
        for (size_t k = 0; k < squares.size(); k++) {
          if (i == k || j == k) continue;

          cv::Point x = getCentre(squares[i]);
          cv::Point y = getCentre(squares[j]);
          cv::Point z = getCentre(squares[k]);
          if (std::fabs(angle(x, y, z)) > eps) continue;

          auto maxSquare = std::max(std::max(cv::contourArea(squares[i]), cv::contourArea(squares[j])), cv::contourArea(squares[k]));
          auto minSquare = std::min(std::min(cv::contourArea(squares[i]), cv::contourArea(squares[j])), cv::contourArea(squares[k]));
          if (sqrt(maxSquare / minSquare) > 1 + eps) continue;

          double xz = distance(x, z);
          double yz = distance(y, z);
          if (xz / yz > 1 + eps || yz / xz > 1 + eps) continue;

          std::vector<cv::Point> tmp = {x, z, y, x + (y - z)};
          double sca = (distance(x, y) + sqrt(sqrt(maxSquare * minSquare)) * 1.55) / distance(x, y);
          cv::Point centre = getCentre(tmp);
          std::vector<cv::Point> scaled;
          for (auto u : tmp) {
            scaled.push_back(centre + (u - centre) * sca);
          }
          if (ans.empty() || cv::contourArea(ans) < cv::contourArea(scaled)) {
            ans = scaled;
          }
        }
      }
    }
    if (ans.size()) {
      break;
    }
  }
  if (ans.size()) {
    biggest_square = ans;
    return true;
  }
  return false;
}

inline bool findLargestSquare(
    const std::vector<std::vector<cv::Point>>& squares,
    std::vector<cv::Point>& biggest_square) {
  if (!squares.size()) {
    std::cout << "No squares detect" << std::endl;
    return false;
  }


  int max_width = 0;
  int max_height = 0;
  int max_square_idx = 0;
  int max_area = 0;
  for (size_t i = 0; i < squares.size(); i++) {
    cv::Rect rectangle = cv::boundingRect(cv::Mat(squares[i]));

    if ((rectangle.width >= max_width) && (rectangle.height >= max_height)) {
      max_width = rectangle.width;
      max_height = rectangle.height;
      max_square_idx = i;
    }
    // if (max_area < cv::contourArea(squares[i])) {
    //   max_area = cv::contourArea(squares[i]);
    //   max_square_idx = i;
    // }
  }

  biggest_square = squares[max_square_idx];
  return true;
}
