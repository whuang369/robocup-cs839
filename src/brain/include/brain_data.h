#pragma once

#include <string>
#include <mutex>
#include <unordered_map>
#include <deque>

#include "locator.h"
#include "team_communication_msg.h"

/**
 * The BrainData class records the data needed by the Brain during decision-making.
 * Currently, multi-threaded read/write issues are not considered, but this may be addressed in the
 * future if necessary.
 */
class BrainData {
 public:
  rclcpp::Time lastSuccessfulLocalizeTime;

  int lastScore = 0;
  int penalty[4];

  /* ------------------------------------ Data Recording ------------------------------------ */

  // Robot position & velocity commands
  Pose2D
      robotPoseToOdom;  // The robot's Pose in the Odom coordinate system, updated via odomCallback

  //   Pose3D visualOdom; // camera's Pose in the camera odom coordinate system (visualOdomCallback)

  Pose2D odomToField;  // The origin of the Odom coordinate system in the Field coordinate system,
                       // can be calibrated using known positions, e.g., by calibration at the start
                       // of the game
  Pose2D robotPoseToField;  // The robot's current position and orientation in the field coordinate
                            // system. The field center is the origin, with the x-axis pointing
                            // towards the opponent's goal (forward), and the y-axis pointing to the
                            // left. The positive direction of theta is counterclockwise.
  bool walking;             // Whether the robot is walking

  // Head position, updated through lowStateCallback
  double headPitch;   // The current head pitch, in radians. 0 is horizontal forward, positive is
                      // downward.
  double headYaw;     // The current head yaw, in radians. 0 is forward, positive is left.
  double headPitchD;  // The current head pitch differential.
  double headYawD;    // The current head yaw differential.

  // Ball
  bool ballDetected = false;  // Whether the camera has detected the ball
  GameObject ball;  // Records the ball's information, including position, bounding box, etc.
  double robotBallAngleToField;  // The angle between the robot's vector to the ball and the X-axis
                                 // in the field coordinate system, (-PI, PI]
  double ballVelocityX;          // The velocity of the ball in the field coordinate system
  double ballVelocityY;          // The velocity of the ball in the field coordinate system
  rclcpp::Time lastOpponentNearBallTime;

  // 起身
  RobotRecoveryState recoveryState = RobotRecoveryState::IS_READY;
  bool isRecoveryAvailable = false;  // 是否可以起身
  int currentRobotModeIndex = -1;
  bool recoveryPerformed = false;  // 是否发送起身命令4
  rclcpp::Time lastRecoveryTime;   // 上次起身的时间
  bool enterDampingPerformed = false;
  bool needManualRelocate = false;

  // Other objects on the field
  std::vector<GameObject> opponents = {};  // Records information about opponent players
  std::vector<GameObject> goalposts = {};  // Records information about goalposts
  std::vector<GameObject> markings = {};   // Records information about field markings

  // Team communication
  std::mutex teamCommunicationMutex;
  std::unordered_map<int, TeamCommunicationMsg> teamMemberMessages;  // Records team messages
  int electedKickerId = 1;
  rclcpp::Time kickerElectionTime;

  // A collection of utility functions
  std::vector<FieldMarker> getMarkers();
  // Convert a Pose from the robot coordinate system to the field coordinate system.
  Pose2D robot2field(const Pose2D &poseToRobot);
  Point robot2field(const Point &pointToRobot);
  // Convert a Pose from the field coordinate system to the robot coordinate system.
  Pose2D field2robot(const Pose2D &poseToField);
  Point field2robot(const Point &pointToField);
};