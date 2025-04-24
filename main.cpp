#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "JsonParser.hpp"
#include "QrDetector.hpp"
#include "Util.hpp"

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void benchmark() {
  std::string pathToFolder = "../img/";
  std::string pathToJson = "annotations.json";
  cv::FileStorage file =
      cv::FileStorage(pathToFolder + pathToJson, cv::FileStorage::READ);
  JsonParser parser = JsonParser(file);
  auto standard = parser.parsePolygon();

  QrDetector::Benchmark(pathToFolder, standard);
}

void detect(const std::string namePhoto) {
  cv::Mat src = cv::imread("../img/" + namePhoto);
  QrDetector detect = QrDetector(src);
  auto points = detect.Detection();
  cv::polylines(src, points, true, cv::Scalar(0, 255, 0), 3);
  if (src.rows > 1080) {
    cv::Mat resizedImage;
    cv::resize(src, resizedImage, cv::Size(src.cols * 1080.0 / src.rows, 1080));
    src = resizedImage;
  }
  normImShow("res", src);
}

int main(int argc, char** argv) {
  cv::CommandLineParser parser(
      argc, argv,
      "{mode           |      | Режим работы: benchmark или detect}"
      "{image @image   |      | Путь к изображению (требуется для режима "
      "detect)}");

  std::string mode = parser.get<std::string>("mode");

  if (mode.empty()) {
    std::cerr << "Ошибка: Не указан режим работы (benchmark или detect)"
              << std::endl;
    parser.printMessage();
    return -1;
  }

  if (mode == "benchmark") {
    std::cout << "Запущен режим benchmark..." << std::endl;
    benchmark();
  } else if (mode == "detect") {
    std::string image_path = parser.get<std::string>("@image");

    if (image_path.empty()) {
      std::cerr
          << "Ошибка: Для режима detect требуется указать путь к изображению"
          << std::endl;
      parser.printMessage();
      return -1;
    }

    std::cout << "Запущен режим detect с изображением: " << image_path
              << std::endl;
    detect(image_path);
  }
  cv::waitKey(-1);
  return 0;
}