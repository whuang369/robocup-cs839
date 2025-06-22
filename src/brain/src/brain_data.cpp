#include "brain_data.h"
#include "utils/math.h"

using std::string;
using std::unordered_map;
using std::vector;

vector<FieldMarker> BrainData::getMarkers() {
  static const unordered_map<string, char> labelToMarker = {
      {"LCross", 'L'},
      {"TCross", 'T'},
      {"XCross", 'X'},
      {"PenaltyPoint", 'P'},
  };

  vector<FieldMarker> res;
  res.reserve(markings.size());

  for (const auto &mark : markings) {
    char markerType = ' ';
    auto it = labelToMarker.find(mark.label);
    if (it != labelToMarker.end()) {
      markerType = it->second;
    }

    res.emplace_back(
        FieldMarker{markerType, mark.posToRobot.x, mark.posToRobot.y, mark.confidence});
  }
  return res;
}

Pose2D BrainData::robot2field(const Pose2D &poseToRobot) {
  Pose2D poseToField;
  transCoord(poseToRobot.x, poseToRobot.y, poseToRobot.theta, robotPoseToField.x,
             robotPoseToField.y, robotPoseToField.theta, poseToField.x, poseToField.y,
             poseToField.theta);
  poseToField.theta = toPInPI(poseToField.theta);
  return poseToField;
}

Pose2D BrainData::field2robot(const Pose2D &poseToField) {
  Pose2D poseToRobot;
  double xfr, yfr, thetafr;  // fr = field to robot
  yfr = sin(robotPoseToField.theta) * robotPoseToField.x -
        cos(robotPoseToField.theta) * robotPoseToField.y;
  xfr = -cos(robotPoseToField.theta) * robotPoseToField.x -
        sin(robotPoseToField.theta) * robotPoseToField.y;
  thetafr = -robotPoseToField.theta;
  transCoord(poseToField.x, poseToField.y, poseToField.theta, xfr, yfr, thetafr, poseToRobot.x,
             poseToRobot.y, poseToRobot.theta);
  return poseToRobot;
}