#pragma once

#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <rcpputils/endian.hpp>

#include <string>
#include <vector>
#include <stdexcept>
#include <regex>

namespace booster_vision {

namespace enc = sensor_msgs::image_encodings;

inline int depthStrToInt(const std::string& depth) {
  if (depth == "8U") return 0;
  if (depth == "8S") return 1;
  if (depth == "16U") return 2;
  if (depth == "16S") return 3;
  if (depth == "32S") return 4;
  if (depth == "32F") return 5;
  if (depth == "64F") return 6;
  throw std::runtime_error("Invalid depth string: " + depth);
}

inline int getCvType(const std::string& encoding) {
  if (encoding == enc::BGR8 || encoding == enc::RGB8) return CV_8UC3;
  if (encoding == enc::MONO8) return CV_8UC1;
  if (encoding == enc::MONO16) return CV_16UC1;
  if (encoding == enc::BGR16 || encoding == enc::RGB16) return CV_16UC3;
  if (encoding == enc::BGRA8 || encoding == enc::RGBA8) return CV_8UC4;
  if (encoding == enc::BGRA16 || encoding == enc::RGBA16) return CV_16UC4;

  // Bayer encodings
  if (encoding.find("BAYER") != std::string::npos) {
    return (encoding.find("16") != std::string::npos) ? CV_16UC1 : CV_8UC1;
  }

  if (encoding == enc::YUV422 || encoding == enc::YUV422_YUY2) return CV_8UC2;

  // Regex generic
  std::cmatch m;
  if (std::regex_match(encoding.c_str(), m, std::regex("(8U|8S|16U|16S|32S|32F|64F)C([0-9]+)"))) {
    return CV_MAKETYPE(depthStrToInt(m[1].str()), std::stoi(m[2].str()));
  }

  if (std::regex_match(encoding.c_str(), m, std::regex("(8U|8S|16U|16S|32S|32F|64F)"))) {
    return CV_MAKETYPE(depthStrToInt(m[1].str()), 1);
  }

  throw std::runtime_error("Unrecognized image encoding: [" + encoding + "]");
}

inline cv::Mat toCVMat(const sensor_msgs::msg::Image& source) {
  int source_type = getCvType(source.encoding);
  int byte_depth = enc::bitDepth(source.encoding) / 8;
  int num_channels = enc::numChannels(source.encoding);

  if (source.step < source.width * byte_depth * num_channels) {
    std::stringstream ss;
    ss << "Image is wrongly formed: step < width * byte_depth * num_channels  or  " << source.step
       << " != " << source.width << " * " << byte_depth << " * " << num_channels;
    throw std::runtime_error(ss.str());
  }

  if (source.height * source.step != source.data.size()) {
    std::stringstream ss;
    ss << "Image is wrongly formed: height * step != size  or  " << source.height << " * "
       << source.step << " != " << source.data.size();
    throw std::runtime_error(ss.str());
  }

  // If the endianness is the same as locally, share the data
  cv::Mat mat(source.height, source.width, source_type, const_cast<uchar*>(&source.data[0]),
              source.step);

  if ((rcpputils::endian::native == rcpputils::endian::big && source.is_bigendian) ||
      (rcpputils::endian::native == rcpputils::endian::little && !source.is_bigendian) ||
      byte_depth == 1) {
    return mat;
  }

  // Otherwise, reinterpret the data as bytes and switch the channels accordingly
  mat = cv::Mat(source.height, source.width, CV_MAKETYPE(CV_8U, num_channels * byte_depth),
                const_cast<uchar*>(&source.data[0]), source.step);
  cv::Mat mat_swap(source.height, source.width, mat.type());

  std::vector<int> fromTo;
  fromTo.reserve(num_channels * byte_depth);
  for (int i = 0; i < num_channels; ++i) {
    for (int j = 0; j < byte_depth; ++j) {
      fromTo.push_back(byte_depth * i + j);
      fromTo.push_back(byte_depth * i + byte_depth - 1 - j);
    }
  }
  cv::mixChannels(std::vector<cv::Mat>(1, mat), std::vector<cv::Mat>(1, mat_swap), fromTo);

  // Interpret mat_swap back as the proper type
  mat_swap.reshape(num_channels);
  return mat_swap;
}

inline cv::Mat toBGRMat(const sensor_msgs::msg::Image& source) {
  cv::Mat input = toCVMat(source);
  if (source.encoding == enc::BGR8) return input;

  cv::Mat converted;
  if (source.encoding == enc::RGB8) {
    cv::cvtColor(input, converted, cv::COLOR_RGB2BGR);
    return converted;
  }
  if (source.encoding == enc::BGRA8) {
    cv::cvtColor(input, converted, cv::COLOR_BGRA2BGR);
    return converted;
  }
  throw std::runtime_error("Unsupported encoding in toBGRMat: " + source.encoding);
}

}  // namespace booster_vision