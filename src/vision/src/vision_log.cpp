#include "booster_vision/vision_log.h"
#include "booster_vision/vision_node.h"

#include <iostream>

#include <opencv2/opencv.hpp>

namespace booster_vision {

VisionLog::VisionLog(VisionNode* visionNode, bool enable, const std::string& rerunServerAddr,
                     const std::string& logId)
    : enabled(enable), visionNode(visionNode), rerunLog("robocup_vision_" + logId) {
  if (!enabled) return;

  rerun::Error err = rerunLog.connect_grpc(rerunServerAddr);
  if (err.is_err()) {
    std::cerr << "VisionLog: Failed to connect to rerun: " << err.description << std::endl;
    enabled = false;
    return;
  }
}

void VisionLog::setTimeNow() {
  if (enabled) setTimeSeconds(visionNode->get_clock()->now().seconds());
}

void VisionLog::setTimeSeconds(double seconds) {
  if (enabled) rerunLog.set_time_seconds("time", seconds);
}

void VisionLog::logDetections(const cv::Mat& color, const Pose& p_eye2base,
                              const vision_interface::msg::Detections& detection_msg) {
  if (!enabled) return;

  setTimeSeconds(detection_msg.header.stamp.sec +
                 static_cast<double>(detection_msg.header.stamp.nanosec) * 1e-9);

  const auto& detections = detection_msg.detected_objects;

  std::vector<uint8_t> compressed_image;
  cv::imencode(".jpg", color, compressed_image, {cv::IMWRITE_JPEG_QUALITY, 10});
  log("image/color", rerun::EncodedImage::from_bytes(compressed_image));

  static std::map<std::string, rerun::Color> detectColorMap = {
      {"Ball", rerun::Color(0xFFFFFFFF)},          // White
      {"LCross", rerun::Color(0xFFFF00FF)},        // Yellow
      {"TCross", rerun::Color(0x00FF00FF)},        // Bright Green
      {"XCross", rerun::Color(0x00FFFFFF)},        // Cyan
      {"PenaltyPoint", rerun::Color(0x87CEFAFF)},  // Light Sky Blue
      {"Person", rerun::Color(0xFF69B4FF)},        // Hot Pink
      {"Goalpost", rerun::Color(0xFFA500FF)},      // Orange
      {"Opponent", rerun::Color(0xFF4500FF)},      // Orange-Red
  };

  std::vector<rerun::Vec2D> mins, sizes;
  std::vector<rerun::Vec3D> positions, positions_projected;
  std::vector<rerun::Text> labels;
  std::vector<rerun::Color> colors;
  mins.reserve(detections.size());
  sizes.reserve(detections.size());
  positions.reserve(detections.size());
  positions.reserve(detections.size());
  labels.reserve(detections.size());
  colors.reserve(detections.size());

  // TODO: check positions valid (size() > 0)
  for (const auto& obj : detections) {
    std::ostringstream oss;
    // clang-format off
    oss << obj.label << " "
        << "c:" << std::fixed << std::setprecision(2) << obj.confidence
        << " x:" << std::fixed << std::setprecision(2) << obj.position[0]
        << " y:" << std::fixed << std::setprecision(2) << obj.position[1]
        << " z:" << std::fixed << std::setprecision(2) << obj.position[2];
    // clang-format on

    labels.emplace_back(rerun::Text(oss.str()));
    positions.emplace_back(rerun::Vec3D(obj.position[0], obj.position[1], obj.position[2]));
    positions_projected.emplace_back(rerun::Vec3D(
        obj.position_projection[0], obj.position_projection[1], obj.position_projection[2]));
    mins.emplace_back(rerun::Vec2D(obj.xmin, obj.ymin));
    sizes.emplace_back(rerun::Vec2D(obj.xmax - obj.xmin, obj.ymax - obj.ymin));
    colors.emplace_back(detectColorMap[obj.label]);
  }

  log("image/detections",
      rerun::Boxes2D::from_mins_and_sizes(mins, sizes).with_labels(labels).with_colors(colors));

  log("base/detections",
      rerun::Points3D(positions).with_colors(colors).with_labels(labels).with_radii(0.3f));

  log("base/detections_projected", rerun::Points3D(positions_projected)
                                       .with_colors(colors)
                                       .with_labels(labels)
                                       .with_radii(0.3f));
  // frames
  log("base/base", rerun::Transform3D(rerun::components::Translation3D(0.0f, 0.0f, 0.0f),
                                      rerun::Quaternion::from_xyzw(0.0, 0.0, 0.0, 1.0))
                       .with_axis_length(1.0f));

  std::vector<float> eye_t = p_eye2base.getTranslationVec();
  std::vector<double> eye_q = p_eye2base.getQuaternionVec();
  log("base/eye",
      rerun::Transform3D(rerun::components::Translation3D(eye_t[0], eye_t[1], eye_t[2]),
                         rerun::Quaternion::from_xyzw(eye_q[0], eye_q[1], eye_q[2], eye_q[3]))
          .with_axis_length(1.0f));
}

}  // namespace booster_vision