#pragma once

#include <string>
#include <ostream>

#include "types.h"
#include "utils/math.h"

/**
 * Stores configuration values required by the Brain. These values should be confirmed during
 * initialization and remain read-only during the robot's decision-making process. Values that need
 * to change during the decision process should be placed in BrainData.
 *
 * Note:
 * 1. The configuration file will be read from config/config.yaml.
 * 2. If config/config_local.yaml exists, its values will override those in config/config.yaml.
 */

class BrainConfig {
 public:
  // ---------- start config from config.yaml ---------------------------------------------
  // These variables are the raw values directly read from the configuration file.
  // If new configurations are added to the configuration file, corresponding variables should be
  // added here to store them. These values will be overwritten in BrainNode, so even if a
  // configuration is not explicitly defined in config.yaml, the default values here will not take
  // effect. The actual default values should be configured in the BrainNode's declare_parameter.
  int teamId;                    // game.team_id
  int playerId;                  // game.player_id
  std::string fieldType;         // game.field_type  "adult_size"(14*9) | "kid_size" (9*6)
  std::string playerRole;        // game.player_role   "striker" | "goal_keeper"
  std::string playerAttackSide;  // game.player_attack_side  "left" | "right"
  float playerStartPos;          // game.player_start_pos  [-1.0, 1.0]  // -1.0: left, 1.0: right

  std::vector<std::string> discoveryIpList;
  std::vector<std::string> gameControllerIpList;
  bool enableCom;

  double robotHeight;      // robot.robot_height
  double robotOdomFactor;  // robot.odom_factor odom
  double vxFactor;   // robot.vx_factor fix the issue where the actual vx is larger than the command
  double yawOffset;  // robot.yaw_offset fix the issue of leftward bias during distance measurement
  std::string joystick;  // robot.joystick "logicall" | "beitong"

  double camPixX;  // image.width
  double camPixY;  // image.height

  std::string visualOdomTopic;  // visual_odom.topic the topic of the visual odometry

  bool rerunLogEnable;             // rerunLog.enable  Whether to enable rerunLog
  std::string rerunLogServerAddr;  // rerunLog.server_addr  rerunLog address

  std::string treeFilePath;  //  It is no longer placed in config.yaml; the path to the
                             //  behavior-tree file is now specified in launch.py.
  // ----------  end config from config.yaml ---------------------------------------------

  // game parameters
  FieldDimensions fieldDimensions;

  // Camera angle
  double camAngleX = deg2rad(90);
  double camAngleY = deg2rad(65);

  // Head rotation soft limit
  double headYawLimitLeft = 1.1;
  double headYawLimitRight = -1.1;
  double headPitchLimitUp = 0.0;

  // Speed limit
  double vxLimit = 1.2;
  double vyLimit = 0.4;
  double vthetaLimit = 1.5;

  // Strategy parameters
  double safeDist = 2.0;  // Safety distance for collision detection. If the distance is smaller
                          // than this value, a collision is considered.
  double goalPostMargin = 0.4;
  // Calculate the margin of the goalpost, used to compute the angle of the goalpost. The larger the
  // margin, the smaller the goal appears during calculations. During a touch event, this margin is
  // different and is typically smaller.
  double goalPostMarginForTouch = 0.1;
  double memoryLength = 3.0;  // The number of seconds during which the ball is not visible, after
                              // which it is considered lost.

  // After Brain fills in the parameters, it calls handle() to process the parameters (such as
  // calibration, calculations, etc.),
  void handle();

  // Output the configuration information to the specified output stream (for debugging).
  void print(ostream &os);
};