#pragma once

#include <string>

#include <opencv2/core.hpp>
#include <rerun.hpp>

#include "booster_vision/base/pose.h"
#include "vision_interface/msg/detections.hpp"

namespace booster_vision {

class VisionNode;

class VisionLog {
 public:
  VisionLog(VisionNode* visionNode, bool enable, const std::string& rerunServerAddr,
            const std::string& logId);

  void setTimeNow();
  void setTimeSeconds(double seconds);

  // Expose the same interface as rerun::RecordingStream
  template <typename... Ts>
  inline void log(std::string_view entity_path, const Ts&... archetypes_or_collections) const {
    if (enabled) rerunLog.log(entity_path, archetypes_or_collections...);
  }

  void logDetections(const cv::Mat& color, const Pose& p_eye2base,
                     const vision_interface::msg::Detections& detection_msg);

  inline bool isEnabled() { return enabled; }

 private:
  bool enabled;
  VisionNode* visionNode;
  rerun::RecordingStream rerunLog;
};

}  // namespace booster_vision