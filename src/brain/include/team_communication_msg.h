#pragma once

#include "types.h"

#define VALIDATION_COMMUNICATION 31201
#define VALIDATION_DISCOVERY 41202
struct TeamCommunicationMsg {
  int validation = VALIDATION_COMMUNICATION;  // validate msg, to determine if it's sent by us.
  int communicationId;
  int teamId;
  int playerId;
  // TODO: You need to add something you want to send to teammates
  rclcpp::Time timePoint;
  GameObject ball;
  Pose2D robotPoseToField;
};

struct TeamDiscoveryMsg {
  int validation = VALIDATION_DISCOVERY;  // validate msg, to determine if it's sent by us.
  int communicationId;
  int teamId;
  int playerId;
};
