#include "booster_vision/vision_log.h"
#include "booster_vision/vision_node.h"

#include <iostream>

namespace booster_vision {

VisionLog::VisionLog(VisionNode* visionNode, bool enable, const std::string& rerunServerAddr)
    : enabled(enable), visionNode(visionNode), rerunLog("robocup_vision") {
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

}  // namespace booster_vision