#include <iostream>
#include <string>

#include <rerun.hpp>
#include <opencv2/opencv.hpp>

#include "brain.h"
#include "utils/print.h"
#include "utils/math.h"
#include "joy_msg.h"

#include "Math/Pose2f.h"

using std::bind;
using std::string;
using std::vector;
using std::placeholders::_1;

Brain::Brain() : rclcpp::Node("brain_node") {
  // Note that the parameters must be declared here first, otherwise they cannot be read in the
  // program either.
  declare_parameter<int>("game.team_id", 0);
  declare_parameter<int>("game.player_id", 29);
  declare_parameter<string>("game.field_type", "");

  declare_parameter<string>("game.player_role", "");
  declare_parameter<string>("game.player_attack_side", "");
  declare_parameter<float>("game.player_start_pos", 0.0f);

  declare_parameter<vector<string>>("discovery_ip_list", {});
  declare_parameter<vector<string>>("game_controller_ip_list", {});
  declare_parameter<bool>("enable_com", true);

  declare_parameter<double>("robot.robot_height", 1.0);
  declare_parameter<double>("robot.odom_factor", 1.0);
  declare_parameter<double>("robot.vx_factor", 0.95);
  declare_parameter<double>("robot.yaw_offset", 0.1);
  declare_parameter<string>("robot.joystick", "");

  declare_parameter<int>("image.width", 960);
  declare_parameter<int>("image.height", 540);

  declare_parameter<string>("visual_odom.topic", "");

  declare_parameter<bool>("rerunLog.enable", false);
  declare_parameter<string>("rerunLog.server_addr", "");
  // The tree_file_path is configured in launch.py and not placed in config.yaml.
  declare_parameter<string>("tree_file_path", "");
}

void Brain::init() {
  // Make sure to load the configuration first, and then the config can be used.
  config = std::make_shared<BrainConfig>();
  loadConfig();

  log = std::make_shared<BrainLog>(this);
  log->prepare();

  data = std::make_shared<BrainData>();
  self_locator = std::make_shared<SelfLocator>(this, config->fieldDimensions);
  self_locator->init(config->fieldDimensions, config->playerAttackSide, config->playerStartPos);

  tree = std::make_shared<BrainTree>(this);
  client = std::make_shared<RobotClient>(this);
  communication = std::make_shared<BrainCommunication>(this);

  tree->init();

  client->init();

  communication->initUDPBroadcast();

  data->lastSuccessfulLocalizeTime = get_clock()->now();

  joySubscription = create_subscription<sensor_msgs::msg::Joy>(
      "/joy", 10, bind(&Brain::joystickCallback, this, _1));
  gameControlSubscription = create_subscription<game_controller_interface::msg::GameControlData>(
      "/robocup/game_controller", 1, bind(&Brain::gameControlCallback, this, _1));
  detectionsSubscription = create_subscription<vision_interface::msg::Detections>(
      "/booster_vision/detection", 1, bind(&Brain::detectionsCallback, this, _1));
  odometerSubscription = create_subscription<booster_interface::msg::Odometer>(
      "/odometer_state", 1, bind(&Brain::odometerCallback, this, _1));
  lowStateSubscription = create_subscription<booster_interface::msg::LowState>(
      "/low_state", 1, bind(&Brain::lowStateCallback, this, _1));
  visualOdomSubscription = create_subscription<geometry_msgs::msg::PoseStamped>(
      config->visualOdomTopic, 1, bind(&Brain::visualOdomCallback, this, _1));
  headPoseSubscription = create_subscription<geometry_msgs::msg::Pose>(
      "/head_pose", 1, bind(&Brain::headPoseCallback, this, _1));
  recoveryStateSubscription = create_subscription<booster_interface::msg::RawBytesMsg>(
      "fall_down_recovery_state", 1, bind(&Brain::recoveryStateCallback, this, _1));
}

void Brain::loadConfig() {
  get_parameter("game.team_id", config->teamId);
  get_parameter("game.player_id", config->playerId);
  get_parameter("game.field_type", config->fieldType);
  get_parameter("game.player_role", config->playerRole);
  get_parameter("game.player_attack_side", config->playerAttackSide);
  get_parameter("game.player_start_pos", config->playerStartPos);

  get_parameter("discovery_ip_list", config->discoveryIpList);
  get_parameter("game_controller_ip_list", config->gameControllerIpList);
  get_parameter("enable_com", config->enableCom);

  get_parameter("robot.robot_height", config->robotHeight);
  get_parameter("robot.odom_factor", config->robotOdomFactor);
  get_parameter("robot.vx_factor", config->vxFactor);
  get_parameter("robot.yaw_offset", config->yawOffset);
  get_parameter("robot.joystick", config->joystick);

  int imgWidth, imgHeight;
  get_parameter("image.width", imgWidth);
  get_parameter("image.height", imgHeight);
  config->camPixX = static_cast<double>(imgWidth);
  config->camPixY = static_cast<double>(imgHeight);

  get_parameter("visual_odom.topic", config->visualOdomTopic);

  get_parameter("rerunLog.enable", config->rerunLogEnable);
  get_parameter("rerunLog.server_addr", config->rerunLogServerAddr);

  get_parameter("tree_file_path", config->treeFilePath);

  // handle the parameters
  config->handle();

  // debug after handle the parameters
  std::ostringstream oss;
  config->print(oss);
  prtDebug(oss.str());
}

/**
 * will be called in the Ros2 loop
 */
void Brain::tick() {
  updateMemory();
  tree->tick();
}

void Brain::updateMemory() {
  updateBallMemory();

  static Point ballPos;
  static rclcpp::Time kickOffTime;
  if (tree->getEntry<string>("player_role") == "striker" &&
      ((tree->getEntry<string>("gc_game_state") == "SET" &&
        !tree->getEntry<bool>("gc_is_kickoff_side")) ||
       (tree->getEntry<string>("gc_game_sub_state") == "SET" &&
        !tree->getEntry<bool>("gc_is_sub_state_kickoff_side")))) {
    ballPos = data->ball.posToRobot;
    kickOffTime = get_clock()->now();
    tree->setEntry<bool>("wait_for_opponent_kickoff", true);
  } else if (tree->getEntry<bool>("wait_for_opponent_kickoff")) {
    if (norm(data->ball.posToRobot.x - ballPos.x, data->ball.posToRobot.y - ballPos.y) > 0.3 ||
        (get_clock()->now() - kickOffTime).seconds() > 10.0) {
      tree->setEntry<bool>("wait_for_opponent_kickoff", false);
    }
  }
}

void Brain::updateBallMemory() {
  // mark ball as lost if long time no see
  if (get_clock()->now().seconds() - data->ball.timePoint.seconds() > config->memoryLength) {
    // use team member's ball memory if available
    bool teamMemberBallFound = false;
    for (const auto &it : data->teamMemberMessages) {
      if (get_clock()->now().seconds() - it.second.ballTimePoint.seconds() < config->memoryLength) {
        data->ball.timePoint = it.second.ballTimePoint;
        data->ball.posToField = it.second.ballPosToField;
        teamMemberBallFound = true;
        log->log("brain/updateBallMemory",
                 rerun::TextLog(format("Ball found in team member %d", it.second.playerId)));
        break;
      }
    }
    tree->setEntry<bool>("ball_location_known", teamMemberBallFound);
    data->ballDetected = teamMemberBallFound;
    if (!teamMemberBallFound) {
      log->log(
          "brain/updateBallMemory",
          rerun::TextLog(format("Ball Lost! Last time: %.2f, Current time: %.2f`",
                                data->ball.timePoint.seconds(), get_clock()->now().seconds())));
    }
    log->log(
        "field/memball",
        rerun::LineStrips2D({
                                rerun::Collection<rerun::Vec2D>{
                                    {data->ball.posToField.x - 0.2, -data->ball.posToField.y},
                                    {data->ball.posToField.x + 0.2, -data->ball.posToField.y}},
                                rerun::Collection<rerun::Vec2D>{
                                    {data->ball.posToField.x, -data->ball.posToField.y - 0.2},
                                    {data->ball.posToField.x, -data->ball.posToField.y + 0.2}},
                            })
            .with_colors({tree->getEntry<bool>("ball_location_known") ? 0x0000FFFF : 0xFF0000FF})
            .with_radii({0.005})
            .with_draw_order(30));
  }

  data->ball.posToRobot = data->field2robot(data->ball.posToField);

  data->ball.range = std::sqrt(data->ball.posToRobot.x * data->ball.posToRobot.x +
                               data->ball.posToRobot.y * data->ball.posToRobot.y);
  tree->setEntry<double>("ball_range", data->ball.range);
  data->ball.yawToRobot = std::atan2(data->ball.posToRobot.y, data->ball.posToRobot.x);
  data->ball.pitchToRobot = std::asin(config->robotHeight / data->ball.range);
  data->robotBallAngleToField = std::atan2(data->ball.posToField.y - data->robotPoseToField.y,
                                           data->ball.posToField.x - data->robotPoseToField.x);
}

vector<double> Brain::getGoalPostAngles(const double margin, const bool targetOurGoal) {
  double goalX = targetOurGoal ? -config->fieldDimensions.length / 2.0  // our goal
                               : config->fieldDimensions.length / 2.0;  // attack side goal
  double goalYLeft = config->fieldDimensions.goalWidth / 2.0;
  double goalYRight = -config->fieldDimensions.goalWidth / 2.0;

  const double theta_l =
      std::atan2(goalYLeft - margin - data->ball.posToField.y, goalX - data->ball.posToField.x);
  const double theta_r =
      std::atan2(goalYRight + margin - data->ball.posToField.y, goalX - data->ball.posToField.x);

  return {theta_l, theta_r};
}

void Brain::calibrateOdom(double x, double y, double theta) {
  double x_or, y_or, theta_or;  // or = odom to robot
  x_or = -std::cos(data->robotPoseToOdom.theta) * data->robotPoseToOdom.x -
         std::sin(data->robotPoseToOdom.theta) * data->robotPoseToOdom.y;
  y_or = std::sin(data->robotPoseToOdom.theta) * data->robotPoseToOdom.x -
         std::cos(data->robotPoseToOdom.theta) * data->robotPoseToOdom.y;
  theta_or = -data->robotPoseToOdom.theta;

  transCoord(x_or, y_or, theta_or, x, y, theta, data->odomToField.x, data->odomToField.y,
             data->odomToField.theta);

  transCoord(data->robotPoseToOdom.x, data->robotPoseToOdom.y, data->robotPoseToOdom.theta,
             data->odomToField.x, data->odomToField.y, data->odomToField.theta,
             data->robotPoseToField.x, data->robotPoseToField.y, data->robotPoseToField.theta);

  log->log("field/robot_pose_sample",
           rerun::Points2D({{x, -y}, {x + 0.1 * cos(theta), -y - 0.1 * sin(theta)}})
               .with_radii({0.2, 0.1})
               .with_colors({0x00FFFFFF, 0xFF0000FF}));
}

double Brain::msecsSince(rclcpp::Time time) {
  return (this->get_clock()->now() - time).nanoseconds() / 1e6;
}

void Brain::joystickCallback(const sensor_msgs::msg::Joy &msg) {
  JoyMsg joy(msg);

  if (!joy.BTN_LT && !joy.BTN_RT && !joy.BTN_LB && !joy.BTN_RB) {
    if (joy.BTN_B) {
      tree->setEntry<bool>("B_pressed", true);
      prtDebug("B is pressed");
    } else if (!joy.BTN_B && tree->getEntry<bool>("B_pressed")) {
      tree->setEntry<bool>("B_pressed", false);
      prtDebug("B is released");
    }
  } else if (joy.BTN_LT && !joy.BTN_RT && !joy.BTN_LB && !joy.BTN_RB) {
    if (joy.AX_DX || joy.AX_DY) {
      config->vxFactor += 0.01 * joy.AX_DX;
      config->yawOffset += 0.01 * joy.AX_DY;
      prtDebug(format("vxFactor = %.2f  yawOffset = %.2f", config->vxFactor, config->yawOffset));
    }

    if (joy.BTN_X) {
      tree->setEntry<int>("control_state", 1);
      client->setVelocity(0., 0., 0.);
      client->moveHead(0., 0.);
      prtDebug("State => 1: CANCEL");
    } else if (joy.BTN_A) {
      tree->setEntry<int>("control_state", 2);
      tree->setEntry<bool>("odom_calibrated", false);
      prtDebug("State => 2: RECALIBRATE");
    } else if (joy.BTN_B) {
      tree->setEntry<int>("control_state", 3);
      prtDebug("State => 3: ACTION");
    } else if (joy.BTN_Y) {
      string curRole = tree->getEntry<string>("player_role");
      curRole == "striker" ? tree->setEntry<string>("player_role", "goal_keeper")
                           : tree->setEntry<string>("player_role", "striker");
      prtDebug("SWITCH ROLE");
    }
  }
}

void Brain::gameControlCallback(const game_controller_interface::msg::GameControlData &msg) {
  auto lastGameState = tree->getEntry<string>("gc_game_state");
  auto lastGameSubState = tree->getEntry<string>("gc_game_sub_state");
  auto lastGameSubStateType = tree->getEntry<string>("gc_game_sub_state_type");
  auto lastIsUnderPenalty = tree->getEntry<bool>("gc_is_under_penalty");

  vector<string> gameStateMap = {
      "INITIAL",  // Initialization state, players are ready outside the field.
      "READY",    // Ready state, players enter the field and walk to their starting positions.
      "SET",      // Stop action, waiting for the referee machine to issue the instruction to start
                  // the game.
      "PLAY",     // Normal game.
      "END"       // The game is over.
  };
  string gameState = gameStateMap[static_cast<int>(msg.state)];
  tree->setEntry<string>("gc_game_state", gameState);
  bool isKickOffSide = (msg.kick_off_team == config->teamId);
  tree->setEntry<bool>("gc_is_kickoff_side", isKickOffSide);

  string gameSubStateType = static_cast<int>(msg.secondary_state) == 0 ? "NONE" : "FREE_KICK";
  vector<string> gameSubStateMap = {
      "STOP",       // stop the robot
      "GET_READY",  // placement
      "SET"         // waiting for the gameSubStateType to be None
  };
  string gameSubState = gameSubStateMap[static_cast<int>(msg.secondary_state_info[1])];
  tree->setEntry<string>("gc_game_sub_state_type", gameSubStateType);
  tree->setEntry<string>("gc_game_sub_state", gameSubState);
  bool isSubStateKickOffSide = (static_cast<int>(msg.secondary_state_info[0]) == config->teamId);
  tree->setEntry<bool>("gc_is_sub_state_kickoff_side", isSubStateKickOffSide);

  game_controller_interface::msg::TeamInfo myTeamInfo;
  if (msg.teams[0].team_number == config->teamId) {
    myTeamInfo = msg.teams[0];
  } else if (msg.teams[1].team_number == config->teamId) {
    myTeamInfo = msg.teams[1];
  } else {
    log->log("brain/gameControlCallback",
             rerun::TextLog("Received invalid game controller message, team ID not found.")
                 .with_color(rerun::Color(0xFF0000FF)));
    return;
  }

  // TODO: use number_of_warnings, yellow_card_count, red_card_count
  data->penalty[0] = static_cast<int>(myTeamInfo.players[0].penalty);
  data->penalty[1] = static_cast<int>(myTeamInfo.players[1].penalty);
  data->penalty[2] = static_cast<int>(myTeamInfo.players[2].penalty);
  data->penalty[3] = static_cast<int>(myTeamInfo.players[3].penalty);
  double isUnderPenalty = (data->penalty[config->playerId] != 0);
  tree->setEntry<bool>("gc_is_under_penalty", isUnderPenalty);

  int curScore = static_cast<int>(myTeamInfo.score);
  if (curScore > data->lastScore) {
    tree->setEntry<bool>("we_just_scored", true);
    data->lastScore = curScore;
  }
  if (gameState == "SET") {
    tree->setEntry<bool>("we_just_scored", false);
  }

  if (gameState != lastGameState) {
    log->log("brain/gameControlCallback",
             rerun::TextLog(format("Game state changed: %s -> %s; isSubStateKickOffSide: %d",
                                   lastGameState.c_str(), gameState.c_str(),
                                   tree->getEntry<bool>("gc_is_sub_state_kickoff_side")))
                 .with_color(rerun::Color(0x00FF00FF)));  // Green
  }
  if (gameSubState != lastGameSubState) {
    log->log(
        "brain/gameControlCallback",
        rerun::TextLog(
            format(
                "Game sub state changed: %s -> %s; sub state type: %s; isSubStateKickOffSide: %d",
                lastGameSubState.c_str(), gameSubState.c_str(), gameSubStateType.c_str(),
                tree->getEntry<bool>("gc_is_sub_state_kickoff_side")))
            .with_color(rerun::Color(0x0000FFFF)));  // Blue
  }
  if (isUnderPenalty != lastIsUnderPenalty) {
    log->log("brain/gameControlCallback",
             rerun::TextLog(format("Player %d is under penalty: %d", config->playerId,
                                   data->penalty[config->playerId]))
                 .with_color(rerun::Color(0xFFAA00FF)));  // Orange
  }
}

void Brain::detectionsCallback(const vision_interface::msg::Detections &msg) {
  double time = msg.header.stamp.sec + static_cast<double>(msg.header.stamp.nanosec) * 1e-9;

  auto gameObjects = getGameObjects(msg);

  vector<GameObject> balls, goalPosts, persons, robots, obstacles, markings;
  for (const auto &obj : gameObjects) {
    const auto &label = obj.label;

    if (label == "Ball") {
      balls.push_back(obj);
    } else if (label == "Goalpost") {
      goalPosts.push_back(obj);
    } else if (label == "Person") {
      persons.push_back(obj);
      if (tree->getEntry<bool>("treat_person_as_robot")) {
        robots.push_back(obj);
      }
    } else if (label == "Opponent") {
      robots.push_back(obj);
    } else if (label == "LCross" || label == "TCross" || label == "XCross" ||
               label == "PenaltyPoint") {
      markings.push_back(obj);
    }
  }

  detectProcessBalls(balls);
  detectProcessMarkings(markings);
  detectProcessGoalPosts(goalPosts);
  detectProcessRobots(robots);

  // rerun logging
  if (!log->isEnabled()) return;

  // check vision/src/model/detector.cc
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

  auto getColor = [&](const std::string &label) {
    auto it = detectColorMap.find(label);
    return it != detectColorMap.end() ? it->second : rerun::Color(0xFF8080FF);
  };

  // log processed detections
  log->setTimeSeconds(time);
  log->log("field/detections", rerun::Clear::RECURSIVE);

  std::unordered_map<string, int> label_counts;
  for (const auto &marking : data->markings) {
    int count = label_counts[marking.label]++;
    log->log("field/detections/" + marking.label + "_" + std::to_string(count),
             rerun::Points2D({{marking.posToField.x, -marking.posToField.y}})
                 .with_radii(0.05)
                 .with_colors(getColor(marking.label)));
  }
  for (const auto &opponent : data->opponents) {
    int count = label_counts["Opponent"]++;
    log->log("field/detections/opponents_" + std::to_string(count),
             rerun::Points2D({{opponent.posToField.x, -opponent.posToField.y}})
                 .with_radii(0.05)
                 .with_colors(detectColorMap["Opponent"]));
  }
  if (data->ballDetected) {
    log->log("field/detections/ball",
             rerun::Points2D({{data->ball.posToField.x, -data->ball.posToField.y}})
                 .with_radii(0.08)
                 .with_colors(detectColorMap["Ball"]));

    // log mem ball pos
    log->log("field/memball",
             rerun::LineStrips2D({
                                     rerun::Collection<rerun::Vec2D>{
                                         {data->ball.posToField.x - 0.2, -data->ball.posToField.y},
                                         {data->ball.posToField.x + 0.2, -data->ball.posToField.y}},
                                     rerun::Collection<rerun::Vec2D>{
                                         {data->ball.posToField.x, -data->ball.posToField.y - 0.2},
                                         {data->ball.posToField.x, -data->ball.posToField.y + 0.2}},
                                 })
                 .with_colors({0xFFFFFFFF})
                 .with_radii({0.005})
                 .with_draw_order(30));
  }
}

void Brain::odometerCallback(const booster_interface::msg::Odometer &msg) {
  constexpr double odomMoveThreshold = 0.01;

  double newOdomX = msg.x * config->robotOdomFactor;
  double newOdomY = msg.y * config->robotOdomFactor;

  data->walking = (std::abs(newOdomX - data->robotPoseToOdom.x) > odomMoveThreshold ||
                   std::abs(newOdomY - data->robotPoseToOdom.y) > odomMoveThreshold);

  data->robotPoseToOdom.x = newOdomX;
  data->robotPoseToOdom.y = newOdomY;
  data->robotPoseToOdom.theta = msg.theta;

  transCoord(data->robotPoseToOdom.x, data->robotPoseToOdom.y, data->robotPoseToOdom.theta,
             data->odomToField.x, data->odomToField.y, data->odomToField.theta,
             data->robotPoseToField.x, data->robotPoseToField.y, data->robotPoseToField.theta);

  log->setTimeNow();
  log->log("field/robot",
           rerun::Points2D({{data->robotPoseToField.x, -data->robotPoseToField.y},
                            {data->robotPoseToField.x + 0.1 * cos(data->robotPoseToField.theta),
                             -data->robotPoseToField.y - 0.1 * sin(data->robotPoseToField.theta)}})
               .with_radii({0.2, 0.1})
               .with_colors({0xFF6666FF, 0xFF0000FF}));
}

void Brain::visualOdomCallback(const geometry_msgs::msg::PoseStamped &msg) {
  log->setTimeNow();
  log->log("camera/odom",
           rerun::Transform3D(
               rerun::components::Translation3D(msg.pose.position.x, msg.pose.position.y,
                                                msg.pose.position.z),
               rerun::Quaternion::from_wxyz(msg.pose.orientation.w, msg.pose.orientation.x,
                                            msg.pose.orientation.y, msg.pose.orientation.z))
               .with_axis_length(0.1));
}

void Brain::lowStateCallback(const booster_interface::msg::LowState &msg) {
  data->headYaw = msg.motor_state_serial[0].q;
  data->headPitch = msg.motor_state_serial[1].q;
  data->headYawD = msg.motor_state_serial[0].dq;
  data->headPitchD = msg.motor_state_serial[1].dq;
  // prtDebug("Head Yaw dq " + to_string(msg.motor_state_serial[0].dq) + " pitch dq " +
  // to_string(msg.motor_state_serial[1].dq));

  // log->setTimeNow();
  // log->log("low_state_callback/imu/rpy/roll", rerun::Scalar(msg.imu_state.rpy[0]));
  // log->log("low_state_callback/imu/rpy/pitch", rerun::Scalar(msg.imu_state.rpy[1]));
  // log->log("low_state_callback/imu/rpy/yaw", rerun::Scalar(msg.imu_state.rpy[2]));
  // log->log("low_state_callback/imu/acc/x", rerun::Scalar(msg.imu_state.acc[0]));
  // log->log("low_state_callback/imu/acc/y", rerun::Scalar(msg.imu_state.acc[1]));
  // log->log("low_state_callback/imu/acc/z", rerun::Scalar(msg.imu_state.acc[2]));
  // log->log("low_state_callback/imu/gyro/x", rerun::Scalar(msg.imu_state.gyro[0]));
  // log->log("low_state_callback/imu/gyro/y", rerun::Scalar(msg.imu_state.gyro[1]));
  // log->log("low_state_callback/imu/gyro/z", rerun::Scalar(msg.imu_state.gyro[2]));
}

void Brain::headPoseCallback(const geometry_msgs::msg::Pose &msg) {
  // TODO: integrate with VO for sensor update
  // --- for test:
  // if (config->rerunLogEnable) {
  if (false) {
    auto x = msg.position.x;
    auto y = msg.position.y;
    auto z = msg.position.z;

    auto orientation = msg.orientation;

    auto roll = rad2deg(
        std::atan2(2 * (orientation.w * orientation.x + orientation.y * orientation.z),
                   1 - 2 * (orientation.x * orientation.x + orientation.y * orientation.y)));
    auto pitch =
        rad2deg(std::asin(2 * (orientation.w * orientation.y - orientation.z * orientation.x)));
    auto yaw = rad2deg(
        std::atan2(2 * (orientation.w * orientation.z + orientation.x * orientation.y),
                   1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)));

    log->setTimeNow();

    log->log("head_to_base/text",
             rerun::TextLog("x: " + to_string(x) + " y: " + to_string(y) + " z: " + to_string(z) +
                            " roll: " + to_string(roll) + " pitch: " + to_string(pitch) +
                            " yaw: " + to_string(yaw)));
    log->log("head_to_base/x", rerun::Scalar(x));
    log->log("head_to_base/y", rerun::Scalar(y));
    log->log("head_to_base/z", rerun::Scalar(z));
    log->log("head_to_base/roll", rerun::Scalar(roll));
    log->log("head_to_base/pitch", rerun::Scalar(pitch));
    log->log("head_to_base/yaw", rerun::Scalar(yaw));
  }
}

void Brain::recoveryStateCallback(const booster_interface::msg::RawBytesMsg &msg) {
  // uint8_t state; // IS_READY = 0, IS_FALLING = 1, HAS_FALLEN = 2, IS_GETTING_UP = 3,
  // uint8_t is_recovery_available; // 1 for available, 0 for not available
  // 使用 RobotRecoveryState 结构，将msg里面的msg转换为RobotRecoveryState
  try {
    const vector<unsigned char> &buffer = msg.msg;
    RobotRecoveryStateData recoveryState;
    memcpy(&recoveryState, buffer.data(), buffer.size());

    vector<RobotRecoveryState> recoveryStateMap = {
        RobotRecoveryState::IS_READY, RobotRecoveryState::IS_FALLING,
        RobotRecoveryState::HAS_FALLEN, RobotRecoveryState::IS_GETTING_UP};
    this->data->recoveryState = recoveryStateMap[static_cast<int>(recoveryState.state)];
    this->data->isRecoveryAvailable = static_cast<bool>(recoveryState.is_recovery_available);
    this->data->currentRobotModeIndex = static_cast<int>(recoveryState.current_planner_index);

    // cout << "recoveryState: " << static_cast<int>(recoveryState.state) << endl;
    // cout << "recovery is available: " << static_cast<int>(recoveryState.is_recovery_available)
    // << endl; cout << "current planner idx: " <<
    // static_cast<int>(recoveryState.current_planner_index) << endl;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }
}

vector<GameObject> Brain::getGameObjects(const vision_interface::msg::Detections &detections) {
  vector<GameObject> res;

  auto timestamp = detections.header.stamp;

  rclcpp::Time timePoint(timestamp.sec, timestamp.nanosec);

  for (int i = 0; i < detections.detected_objects.size(); i++) {
    auto obj = detections.detected_objects[i];
    GameObject gObj;

    gObj.timePoint = timePoint;
    gObj.label = obj.label;

    gObj.boundingBox.xmax = obj.xmax;
    gObj.boundingBox.xmin = obj.xmin;
    gObj.boundingBox.ymax = obj.ymax;
    gObj.boundingBox.ymin = obj.ymin;
    gObj.confidence = obj.confidence;

    if (obj.position.size() > 0 && !(obj.position[0] == 0 && obj.position[1] == 0)) {
      gObj.posToRobot.x = obj.position[0];
      gObj.posToRobot.y = obj.position[1];
    } else {
      gObj.posToRobot.x = obj.position_projection[0];
      gObj.posToRobot.y = obj.position_projection[1];
    }

    gObj.range = norm(gObj.posToRobot.x, gObj.posToRobot.y);
    gObj.yawToRobot = std::atan2(gObj.posToRobot.y, gObj.posToRobot.x);
    gObj.pitchToRobot = std::atan2(config->robotHeight, gObj.range);

    gObj.posToField = data->robot2field(gObj.posToRobot);
    // transCoord(gObj.posToRobot.x, gObj.posToRobot.y, 0, data->robotPoseToField.x,
    //            data->robotPoseToField.y, data->robotPoseToField.theta, gObj.posToField.x,
    //            gObj.posToField.y, gObj.posToField.z);

    res.push_back(gObj);
  }

  return res;
}

void Brain::detectProcessBalls(const vector<GameObject> &ballObjs) {
  // Parameters
  const double confidenceValve =
      0.35;  // If the confidence is lower than this threshold, it is considered not a ball (note
             // that the target confidence passed in by the detection module is currently all >
             // 0.2).
  const double pitchLimit =
      deg2rad(0);  // When the pitch of the ball relative to the front of the robot (downward is
                   // positive) is lower than this value, it is considered not a ball. (Because
                   // the ball won't be in the sky.)
  const int timeCountThreshold =
      5;  // Only when the ball is detected in consecutive several frames is it considered a ball.
          // This is only used in the ball-finding strategy.
  const unsigned int detectCntThreshold =
      3;  // The maximum count. Only when the target is detected in such a number of frames is it
          // considered that the target is truly identified. (Currently only used for ball
          // detection.)
  const unsigned int diffConfidThreshold =
      4;  // The threshold for the difference times between the tracked ball and the
          // high-confidence ball. After reaching this threshold, the high-confidence ball will be
          // adopted.

  double bestConfidence = 0;
  double minPixDistance = 1.e4;
  int indexRealBall = -1;   // Which ball is considered to be the real one. -1 indicates that no
                            // ball has been detected.
  int indexTraceBall = -1;  // Track the ball according to the pixel distance. -1 indicates that
                            // no target has been tracked.

  // Find the most likely real ball.
  for (int i = 0; i < ballObjs.size(); i++) {
    auto ballObj = ballObjs[i];

    // Judgment: If the confidence is too low, it is considered a false detection.
    if (ballObj.confidence < confidenceValve) continue;

    // Prevent the lights in the sky from being recognized as balls.
    if (ballObj.posToRobot.x < -0.5 || ballObj.posToRobot.x > 10.0) continue;

    // Find the one with the highest confidence among the remaining balls.
    if (ballObj.confidence > bestConfidence) {
      bestConfidence = ballObj.confidence;
      indexRealBall = i;
    }
  }

  if (indexRealBall >= 0) {
    data->ballDetected = true;
    GameObject oldBall = data->ball;
    data->ball = ballObjs[indexRealBall];
    tree->setEntry<bool>("ball_location_known", true);

    double dx = data->ball.posToRobot.x - oldBall.posToRobot.x;
    double dy = data->ball.posToRobot.y - oldBall.posToRobot.y;
    double dist = hypot(dx, dy);
    double dt = (data->ball.timePoint - oldBall.timePoint).seconds();
    if (dt < 1e-4) dt = 1e-4;

    double alpha = 0.02;
    data->ballVelocityX = (1 - alpha) * data->ballVelocityX + alpha * (dx / dt);
    data->ballVelocityY = (1 - alpha) * data->ballVelocityY + alpha * (dy / dt);

  } else {
    data->ballDetected = false;
    data->ball.boundingBox.xmin = 0;
    data->ball.boundingBox.xmax = 0;
    data->ball.boundingBox.ymin = 0;
    data->ball.boundingBox.ymax = 0;
    data->ball.confidence = 0;
  }

  data->robotBallAngleToField = std::atan2(data->ball.posToField.y - data->robotPoseToField.y,
                                           data->ball.posToField.x - data->robotPoseToField.x);

  log->log("detectProcessBalls", rerun::TextLog(format("ballDetected: %d", data->ballDetected)));
}

void Brain::detectProcessMarkings(const vector<GameObject> &markingObjs) {
  constexpr double confidenceValve = 60;
  constexpr double minX = -0.5, maxX = 10.0;

  data->markings.clear();
  data->markings.reserve(markingObjs.size());

  for (const auto &marking : markingObjs) {
    if (marking.confidence < confidenceValve) continue;
    if (marking.posToRobot.x < minX || marking.posToRobot.x > maxX) continue;

    data->markings.push_back(marking);
  }
}

void Brain::detectProcessGoalPosts(const vector<GameObject> &goalpostObjs) {
  constexpr double confidenceValve = 20;
  constexpr double minX = -0.5, maxX = 3.0, maxY = 3.0;

  data->goalposts.clear();
  data->goalposts.reserve(goalpostObjs.size());

  for (const auto &post : goalpostObjs) {
    if (post.confidence < confidenceValve) continue;
    if (post.posToRobot.x < minX || post.posToRobot.x > maxX || post.posToRobot.y > maxY) continue;

    data->goalposts.push_back(post);
  }
}

void Brain::detectProcessRobots(const vector<GameObject> &robotObjs) {
  constexpr double confidenceValve = 20;
  constexpr double minX = -0.5, maxX = 10.0, maxY = 3.0;

  data->opponents.clear();
  data->opponents.reserve(robotObjs.size());

  for (const auto &robot : robotObjs) {
    if (robot.confidence < confidenceValve) continue;
    if (robot.posToRobot.x < minX || robot.posToRobot.x > maxX || robot.posToRobot.y > maxY)
      continue;

    data->opponents.push_back(robot);
  }
}