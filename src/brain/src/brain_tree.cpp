#include <cmath>
#include "brain_tree.h"
#include "brain.h"
#include "utils/math.h"
#include "utils/print.h"
#include "utils/misc.h"
#include "std_msgs/msg/string.hpp"

using std::atan2;
using std::clamp;
using std::copysign;
using std::cos;
using std::fabs;
using std::hypot;
using std::sin;
using std::string;

namespace {

bool isShotBlocked(const double kickDir, const Point &ballPosToField,
                   const std::shared_ptr<BrainData> data) {
  constexpr double MIN_BLOCK_DISTANCE = 1.0;
  constexpr double MAX_KICK_ANGLE = 0.5;

  bool blocked = false;
  for (const auto &opponent : data->opponents) {
    double dx = opponent.posToField.x - ballPosToField.x;
    double dy = opponent.posToField.y - ballPosToField.y;

    double ballOpponentRange = hypot(dx, dy);  // Distance from ball to opponent
    if (ballOpponentRange > MIN_BLOCK_DISTANCE) {
      continue;
    }

    double ballOpponentAngle = atan2(dy, dx);  // Angle from ball to opponent
    double angleDiffRobot = fabs(toPInPI(kickDir - ballOpponentAngle));
    if (angleDiffRobot < MAX_KICK_ANGLE) {  // TODO: proportional to inverse of distance
      blocked = true;
      break;
    }
  }
  return blocked;
}

bool isDribbleBlocked(const Pose2D &robotPose, const Point &ballPosToField,
                      const std::shared_ptr<BrainData> data) {
  constexpr double MIN_DRIBBLE_DISTANCE = 1.0;
  constexpr double MAX_DRIBBLE_ANGLE = 0.5;

  bool blocked = false;
  for (const auto &opponent : data->opponents) {
    double dx = opponent.posToField.x - ballPosToField.x;
    double dy = opponent.posToField.y - ballPosToField.y;

    double ballOpponentRange = hypot(dx, dy);  // Distance from ball to opponent
    if (ballOpponentRange > MIN_DRIBBLE_DISTANCE) {
      continue;
    }

    double ballOpponentAngle = atan2(dy, dx);  // Angle from ball to opponent
    double angleDiffRobot = fabs(toPInPI(data->robotBallAngleToField - ballOpponentAngle));
    if (angleDiffRobot < MAX_DRIBBLE_ANGLE) {
      blocked = true;
      break;
    }
  }
  return blocked;
}

int getNewKickerId(int playerId, double ballGoalDir, const std::shared_ptr<BrainData> data) {
  constexpr double MIN_KICKER_DISTANCE = 2.0;

  // how good the kicker is
  double selfShootAlignment = fabs(toPInPI(ballGoalDir - data->robotBallAngleToField));
  bool selfBlocked = isDribbleBlocked(data->robotPoseToField, data->ball.posToField, data);

  // Store close robots - <playerId, isBlocked, shootAlignment>
  std::vector<std::tuple<int, bool, double>> closeRobots;

  if (data->ball.range < MIN_KICKER_DISTANCE) {
    closeRobots.emplace_back(playerId, selfBlocked, selfShootAlignment);
  }

  // Loop over teammates
  for (const auto &[id, msg] : data->teamMemberMessages) {
    double dx = msg.robotPoseToField.x - data->ball.posToField.x;
    double dy = msg.robotPoseToField.y - data->ball.posToField.y;
    double dist = hypot(dx, dy);

    if (dist > MIN_KICKER_DISTANCE) {
      continue;  // Ignore teammates that are too far away
    }

    double teammateBallDir = atan2(data->ball.posToField.y - msg.robotPoseToField.y,
                                   data->ball.posToField.x - msg.robotPoseToField.x);
    double shootAlignment = fabs(toPInPI(ballGoalDir - teammateBallDir));
    bool blocked = isDribbleBlocked(msg.robotPoseToField, data->ball.posToField, data);

    closeRobots.emplace_back(msg.playerId, blocked, shootAlignment);
  }

  if (closeRobots.empty()) return playerId;  // No close teammates, self is the kicker

  // Rule application
  int kickerId = playerId;

  bool shotBlocked = isShotBlocked(ballGoalDir, data->ball.posToField, data);
  if (!shotBlocked) {
    double bestAlignment = std::numeric_limits<double>::max();
    for (const auto &[id, blocked, alignment] : closeRobots) {
      if (alignment < bestAlignment) {
        bestAlignment = alignment;
        kickerId = id;
      }
    }
  } else {
    // If the shot is blocked, prefer someone with clear robot-ball path and alignment
    double bestAlignment = std::numeric_limits<double>::max();
    bool foundUnblocked = false;

    for (const auto &[id, blocked, alignment] : closeRobots) {
      if (!blocked) {
        foundUnblocked = true;
        if (alignment < bestAlignment) {
          bestAlignment = alignment;
          kickerId = id;
        }
      }
    }

    if (!foundUnblocked) {
      bestAlignment = selfShootAlignment;
      for (const auto &[id, blocked, alignment] : closeRobots) {
        if (alignment < bestAlignment) {
          bestAlignment = alignment;
          kickerId = id;
        }
      }
    }
  }

  return kickerId;
}

bool isOutField(const std::shared_ptr<BrainData> data, const std::shared_ptr<BrainConfig> config) {
  // Re-enter when robot is out of field around goalpost
  constexpr double X_MARGIN = 0.1;
  constexpr double Y_MARGIN = 1.0;
  return (data->robotPoseToField.y < config->fieldDimensions.goalAreaWidth / 2 + Y_MARGIN) &&
         (data->robotPoseToField.y > -config->fieldDimensions.goalAreaWidth / 2 - Y_MARGIN) &&
         ((data->robotPoseToField.x > config->fieldDimensions.length / 2 + X_MARGIN) ||
          (data->robotPoseToField.x < -config->fieldDimensions.length / 2 - X_MARGIN));
}

}  // namespace

/**
 * Here, a macro definition is used to reduce the amount of code in RegisterBuilder.
 * The effect after expanding REGISTER_BUILDER(Test) is as follows:
 * factory.registerBuilder<Test>(  \
 *      "Test",                    \
 *      [this](const string& name, const NodeConfig& config) { return make_unique<Test>(name,
 * config, brain); });
 */
#define REGISTER_BUILDER(Name)                                                                \
  factory.registerBuilder<Name>(#Name, [this](const string &name, const NodeConfig &config) { \
    return make_unique<Name>(name, config, brain);                                            \
  });

void BrainTree::init() {
  BehaviorTreeFactory factory;

  // Action Nodes
  REGISTER_BUILDER(RobotFindBall)
  REGISTER_BUILDER(Chase)
  REGISTER_BUILDER(SimpleChase)
  REGISTER_BUILDER(Adjust)
  REGISTER_BUILDER(Approach)
  REGISTER_BUILDER(Kick)
  REGISTER_BUILDER(StrikerDecide)
  REGISTER_BUILDER(CamTrackBall)
  REGISTER_BUILDER(CamFindBall)
  REGISTER_BUILDER(CamScanField)
  REGISTER_BUILDER(SelfLocate)
  REGISTER_BUILDER(SetVelocity)
  REGISTER_BUILDER(CheckAndStandUp)
  REGISTER_BUILDER(KeepGoal)
  REGISTER_BUILDER(BlockGoal)
  REGISTER_BUILDER(ReEnterField)
  REGISTER_BUILDER(RotateForRelocate)
  REGISTER_BUILDER(MoveToPoseOnField)
  REGISTER_BUILDER(GoalieDecide)
  REGISTER_BUILDER(WaveHand)

  // Action Nodes for debug
  REGISTER_BUILDER(PrintMsg)

  factory.registerBehaviorTreeFromFile(brain->config->treeFilePath);
  tree = factory.createTree("MainTree");

  // init blackboard entry
  initEntry();
}

void BrainTree::initEntry() {
  setEntry<string>("player_role", brain->config->playerRole);
  setEntry<bool>("ball_location_known", false);
  setEntry<bool>("track_ball", true);
  setEntry<bool>("odom_calibrated", false);
  setEntry<string>("decision", "");
  setEntry<string>("defend_decision", "chase");
  setEntry<double>("ball_range", 0);

  setEntry<bool>("robot_find_ball", false);

  setEntry<bool>("gamecontroller_isKickOff", true);
  setEntry<bool>("gamecontroller_isKickOffExecuted", true);

  setEntry<string>("gc_game_state", "");
  setEntry<string>("gc_game_sub_state_type", "NONE");
  setEntry<string>("gc_game_sub_state", "");
  setEntry<bool>("gc_is_kickoff_side", false);
  setEntry<bool>("gc_is_sub_state_kickoff_side", false);
  setEntry<bool>("gc_is_under_penalty", false);

  setEntry<bool>("treat_person_as_robot", false);
  setEntry<int>("control_state", 0);
  setEntry<bool>("B_pressed", false);

  // fallRecovery相关
  setEntry<bool>("should_recalibrate_after_fall_recovery", false);

  setEntry<bool>("we_just_scored", false);
  setEntry<bool>("wait_for_opponent_kickoff", false);
}

void BrainTree::tick() {
  tree.tickOnce();
}

NodeStatus SetVelocity::tick() {
  double x, y, theta;
  vector<double> targetVec;
  getInput("x", x);
  getInput("y", y);
  getInput("theta", theta);

  brain->log->log("tree/SetVelocity",
                  rerun::TextLog(format("x: %.2f  y: %.2f  theta: %.2f", x, y, theta)));

  auto res = brain->client->setVelocity(x, y, theta);
  return NodeStatus::SUCCESS;
}

NodeStatus CamTrackBall::tick() {
  const double pixTolerance = 10;
  const double smoother = 2.0;

  // handle motion blur while moving head
  bool recentlySeen =
      brain->get_clock()->now().seconds() - brain->data->ball.timePoint.seconds() < 0.5;

  double pitch, yaw;
  if (brain->data->ballDetected) {
    double deltaX = mean(brain->data->ball.boundingBox.xmax, brain->data->ball.boundingBox.xmin) -
                    brain->config->camPixX * 0.5;
    double deltaY = mean(brain->data->ball.boundingBox.ymax, brain->data->ball.boundingBox.ymin) -
                    brain->config->camPixY * 0.8;

    if (std::fabs(deltaX) < pixTolerance && std::fabs(deltaY) < pixTolerance) {
      return NodeStatus::SUCCESS;
    }

    double deltaYaw = deltaX / brain->config->camPixX * brain->config->camAngleX / smoother;
    double deltaPitch = deltaY / brain->config->camPixY * brain->config->camAngleY / smoother;

    pitch = brain->data->headPitch + deltaPitch;
    yaw = brain->data->headYaw - deltaYaw;
  } else if (recentlySeen) {
    pitch = brain->data->headPitch;
    yaw = brain->data->headYaw;
  } else {
    pitch = brain->data->ball.pitchToRobot;
    yaw = brain->data->ball.yawToRobot;
  }

  brain->client->moveHead(pitch, yaw);

  brain->log->log("CamTrackBall", rerun::TextLog(format("ballDetected: %d  pitch: %.2f  yaw: %.2f",
                                                        brain->data->ballDetected, pitch, yaw)));
  return NodeStatus::SUCCESS;
}

CamFindBall::CamFindBall(const string &name, const NodeConfig &config, Brain *_brain)
    : SyncActionNode(name, config), brain(_brain) {
  double lowPitch = 0.8;
  double highPitch = 0.3;
  double leftYaw = 0.55;
  double rightYaw = -0.55;

  _cmdSequence[0][0] = lowPitch;
  _cmdSequence[0][1] = leftYaw;
  _cmdSequence[1][0] = lowPitch;
  _cmdSequence[1][1] = 0;
  _cmdSequence[2][0] = lowPitch;
  _cmdSequence[2][1] = rightYaw;
  _cmdSequence[3][0] = highPitch;
  _cmdSequence[3][1] = rightYaw;
  _cmdSequence[4][0] = highPitch;
  _cmdSequence[4][1] = 0;
  _cmdSequence[5][0] = highPitch;
  _cmdSequence[5][1] = leftYaw;

  _cmdIndex = 0;
  _cmdIntervalMSec = 800;
  _cmdRestartIntervalMSec = 50000;
  _timeLastCmd = brain->get_clock()->now();
}

NodeStatus CamFindBall::tick() {
  if (brain->data->ballDetected) return NodeStatus::SUCCESS;

  bool isTurning = brain->tree->getEntry<bool>("robot_find_ball");

  auto curTime = brain->get_clock()->now();
  auto timeSinceLastCmd = (curTime - _timeLastCmd).nanoseconds() / 1e6;

  double intervalThreshold = isTurning ? _cmdIntervalMSec * 2 : _cmdIntervalMSec;
  if (timeSinceLastCmd < intervalThreshold) return NodeStatus::SUCCESS;

  if (timeSinceLastCmd > _cmdRestartIntervalMSec) {
    _cmdIndex = 0;
  } else {
    _cmdIndex = (_cmdIndex + 1) % (sizeof(_cmdSequence) / sizeof(_cmdSequence[0]));
  }

  // only move pitch when we turn in place (RobotFindBall)
  double pitch = _cmdSequence[_cmdIndex][0];
  double yaw = isTurning ? 0.0 : _cmdSequence[_cmdIndex][1];

  brain->client->moveHead(pitch, yaw);
  _timeLastCmd = curTime;
  return NodeStatus::SUCCESS;
}

NodeStatus CamScanField::tick() {
  auto sec = brain->get_clock()->now().seconds();
  auto msec = static_cast<unsigned long long>(sec * 1000);
  double lowPitch, highPitch, leftYaw, rightYaw;
  getInput("low_pitch", lowPitch);
  getInput("high_pitch", highPitch);
  getInput("left_yaw", leftYaw);
  getInput("right_yaw", rightYaw);
  int msecCycle;
  getInput("msec_cycle", msecCycle);

  int cycleTime = msec % msecCycle;
  double pitch = cycleTime > (msecCycle / 2.0) ? lowPitch : highPitch;
  double yaw = cycleTime < (msecCycle / 2.0)
                   ? (leftYaw - rightYaw) * (2.0 * cycleTime / msecCycle) + rightYaw
                   : (leftYaw - rightYaw) * (2.0 * (msecCycle - cycleTime) / msecCycle) + rightYaw;

  // int i;
  // for (i = 1; i <= 6; i++)
  //   	if (cycleTime < (msecCycle * i / 6))
  //         break;
  // if (i > 3) i = 7 - i;
  // double yaw;
  // if (i == 1) yaw = rightYaw; else if (i == 2) yaw = 0; else yaw = leftYaw;

  brain->client->moveHead(pitch, yaw);
  return NodeStatus::SUCCESS;
}

NodeStatus Chase::tick() {
  if (!brain->tree->getEntry<bool>("ball_location_known")) {
    brain->client->setVelocity(0, 0, 0);
    return NodeStatus::SUCCESS;
  }
  double vxLimit, vyLimit, vthetaLimit, dist;
  getInput("vx_limit", vxLimit);
  getInput("vy_limit", vyLimit);
  getInput("vtheta_limit", vthetaLimit);
  getInput("dist", dist);

  double ballRange = brain->data->ball.range;
  double ballYaw = brain->data->ball.yawToRobot;

  Pose2D target_f, target_r;
  if (brain->data->robotPoseToField.x - brain->data->ball.posToField.x >
      (_state == "chase" ? 1.0 : 0.0)) {
    _state = "circle_back";

    target_f.x = brain->data->ball.posToField.x - dist;

    if (brain->data->robotPoseToField.y > brain->data->ball.posToField.y - _dir)
      _dir = 1.0;
    else
      _dir = -1.0;

    target_f.y = brain->data->ball.posToField.y + _dir * dist;
  } else {  // chase
    _state = "chase";
    target_f.x = brain->data->ball.posToField.x - dist;
    target_f.y = brain->data->ball.posToField.y;
  }

  target_r = brain->data->field2robot(target_f);

  double vx = target_r.x;
  double vy = target_r.y;
  double vtheta = ballYaw * 2.0;

  double linearFactor = 1 / (1 + exp(3 * (ballRange * fabs(ballYaw)) - 3));
  vx *= linearFactor;
  vy *= linearFactor;

  vx = cap(vx, vxLimit, -vxLimit);
  vy = cap(vy, vyLimit, -vyLimit);
  vtheta = cap(vtheta, vthetaLimit, -vthetaLimit);

  brain->client->setVelocity(vx, vy, vtheta, false, false, false);
  return NodeStatus::SUCCESS;
}

NodeStatus SimpleChase::tick() {
  double stopDist, stopAngle, vyLimit, vxLimit;
  getInput("stop_dist", stopDist);
  getInput("stop_angle", stopAngle);
  getInput("vx_limit", vxLimit);
  getInput("vy_limit", vyLimit);

  if (!brain->tree->getEntry<bool>("ball_location_known")) {
    brain->client->setVelocity(0, 0, 0);
    return NodeStatus::SUCCESS;
  }

  double vx = brain->data->ball.posToRobot.x;
  double vy = brain->data->ball.posToRobot.y;
  double vtheta = brain->data->ball.yawToRobot * 2.0;

  double linearFactor =
      1 / (1 + exp(3 * (brain->data->ball.range * fabs(brain->data->ball.yawToRobot)) - 3));
  vx *= linearFactor;
  vy *= linearFactor;

  vx = cap(vx, vxLimit, -0.1);
  vy = cap(vy, vyLimit, -vyLimit);

  if (brain->data->ball.range < stopDist) {
    vx = 0;
    vy = 0;
  }

  brain->client->setVelocity(vx, vy, vtheta, false, false, false);
  return NodeStatus::SUCCESS;
}

NodeStatus Adjust::tick() {
  if (!brain->tree->getEntry<bool>("ball_location_known")) {
    return NodeStatus::SUCCESS;
  }

  double turnThreshold, vxLimit, vyLimit, vthetaLimit, maxRange, minRange;
  getInput("turn_threshold", turnThreshold);
  getInput("vx_limit", vxLimit);
  getInput("vy_limit", vyLimit);
  getInput("vtheta_limit", vthetaLimit);
  getInput("max_range", maxRange);
  getInput("min_range", minRange);
  string position;
  getInput("position", position);

  double vx = 0, vy = 0, vtheta = 0;
  double kickDir =
      (position == "defense")
          ? atan2(brain->data->ball.posToField.y,
                  brain->data->ball.posToField.x + brain->config->fieldDimensions.length / 2)
          : atan2(-brain->data->ball.posToField.y,
                  brain->config->fieldDimensions.length / 2 - brain->data->ball.posToField.x);
  double dir_rb_f = brain->data->robotBallAngleToField;
  double deltaDir = toPInPI(kickDir - dir_rb_f);
  double dir = deltaDir > 0 ? -1.0 : 1.0;
  double ballRange = brain->data->ball.range;
  double ballYaw = brain->data->ball.yawToRobot;

  double s = 0.4;
  double r = 0.8;
  vx = -s * dir * sin(ballYaw);
  if (ballRange > maxRange) vx += 0.1;
  if (ballRange < maxRange) vx -= 0.1;
  vy = s * dir * cos(ballYaw);
  vtheta = (ballYaw - dir * s) / r;

  vx = cap(vx, vxLimit, -vxLimit);
  vy = cap(vy, vyLimit, -vyLimit);
  vtheta = cap(vtheta, vthetaLimit, -vthetaLimit);

  brain->client->setVelocity(vx, vy, vtheta);
  return NodeStatus::SUCCESS;
}

NodeStatus Approach::tick() {
  if (brain->tree->getEntry<bool>("ball_location_known") == false) {
    brain->log->log("tree/Approach", rerun::TextLog("Ball location not known, stopping approach."));
    brain->client->setVelocity(0, 0, 0);
    return NodeStatus::SUCCESS;
  }

  // Read input parameters
  double kickDir, vxLimit, vyLimit, vthetaLimit, maxRange, minRange;
  getInput("kick_dir", kickDir);
  getInput("vx_limit", vxLimit);
  getInput("vy_limit", vyLimit);
  getInput("vtheta_limit", vthetaLimit);
  getInput("max_range", maxRange);
  getInput("min_range", minRange);

  // Variables for the state machine
  double kickVecX = cos(kickDir);
  double kickVecY = sin(kickDir);

  Pose2D robotPose = brain->data->robotPoseToField;
  Point ballPos = brain->data->ball.posToField;

  double targetOffset = (maxRange + minRange) / 2.0;
  Point2D targetPos = {ballPos.x - targetOffset * kickVecX, ballPos.y - targetOffset * kickVecY};
  Point2D toTarget = {targetPos.x - robotPose.x, targetPos.y - robotPose.y};
  double robotTargetAngle = atan2(toTarget.y, toTarget.x);
  double ballRange = brain->data->ball.range;
  double facingAngle = toPInPI(kickDir - brain->data->robotBallAngleToField);
  double approachAngle = toPInPI(kickDir - robotTargetAngle);

  // Determine the phase
  constexpr double GOOD_FACING_ANGLE = 1.0;
  constexpr double BAD_FACING_ANGLE = 1.3;
  constexpr double GOOD_APPROACH_ANGLE = 1.5;
  constexpr double BAD_APPROACH_ANGLE = 1.8;
  bool enteredKickZone = (ballRange < maxRange) && (fabs(facingAngle) < GOOD_FACING_ANGLE);
  bool exitedKickZone = (ballRange > maxRange + 0.2) || (fabs(facingAngle) > BAD_FACING_ANGLE);
  bool goodApproach = (fabs(approachAngle) < GOOD_APPROACH_ANGLE);
  bool badApproach = (fabs(approachAngle) > BAD_APPROACH_ANGLE);

  ApproachPhase newPhase = _phase;
  if (_phase == ApproachPhase::Adjust && exitedKickZone) {
    newPhase = goodApproach ? ApproachPhase::Approach : ApproachPhase::WrapAround;
  } else if (enteredKickZone) {
    newPhase = ApproachPhase::Adjust;
  } else if (_phase == ApproachPhase::Approach && badApproach) {
    newPhase = ApproachPhase::WrapAround;
  } else if (_phase == ApproachPhase::WrapAround && goodApproach) {
    newPhase = ApproachPhase::Approach;
  }

  // Update the phase if it has changed
  if (newPhase != _phase) {
    string msg = "Phase changed: " + std::to_string(static_cast<int>(_phase)) + " -> " +
                 std::to_string(static_cast<int>(newPhase));
    brain->log->log("approach", rerun::TextLog(msg));
    _phase = newPhase;
  }

  // Set the velocity based on the state
  double vx = 0, vy = 0, vtheta = 0;
  constexpr double TURNING_GAIN = 0.8;
  constexpr double ADJUST_SPEED_FACTOR = 0.4;
  constexpr double GUIDE_DISTANCE = 1.0;
  constexpr double BLEND_RANGE = 2.0;
  constexpr double OPPOSITE_FACING_ANGLE = 2.5;

  if (_phase == ApproachPhase::Adjust) {
    double dir = (fabs(facingAngle) < 0.02) ? 0.0 : (facingAngle > 0 ? -1.0 : 1.0);
    double ballYaw = brain->data->ball.yawToRobot;
    vx = -ADJUST_SPEED_FACTOR * dir * sin(ballYaw);
    if (ballRange > maxRange) vx += 0.1;
    if (ballRange < minRange) vx -= 0.1;
    vy = ADJUST_SPEED_FACTOR * dir * cos(ballYaw);
    vtheta = (ballYaw - dir * ADJUST_SPEED_FACTOR) / TURNING_GAIN;
  } else {
    Point2D guidePos;
    double alpha = 1.0;

    if (_phase == ApproachPhase::Approach) {
      // Project onto kick direction
      double proj = toTarget.x * kickVecX + toTarget.y * kickVecY;
      guidePos.x = targetPos.x - clamp(proj, 0.0, GUIDE_DISTANCE) * kickVecX;
      guidePos.y = targetPos.y - clamp(proj, 0.0, GUIDE_DISTANCE) * kickVecY;

      double lateralDist = hypot(toTarget.x - proj * kickVecX, toTarget.y - proj * kickVecY);
      alpha = clamp(lateralDist / BLEND_RANGE, 0.0, 1.0);
    } else if (_phase == ApproachPhase::WrapAround) {
      // Project onto line perpendicular to kick direction
      double proj = toTarget.x * kickVecY - toTarget.y * kickVecX;
      guidePos.x = targetPos.x - copysign(GUIDE_DISTANCE, proj) * kickVecY;
      guidePos.y = targetPos.y + copysign(GUIDE_DISTANCE, proj) * kickVecX;

      double lateralDist = hypot(toTarget.x - proj * kickVecY, toTarget.y + proj * kickVecX);
      alpha = clamp(lateralDist / BLEND_RANGE, 0.0, 1.0);
      if (fabs(proj) < GUIDE_DISTANCE && fabs(facingAngle) > OPPOSITE_FACING_ANGLE) alpha = 1.0;
    }

    // Get the carrot position by interpolating between the guide position and the target position
    Point2D carrotPos = {alpha * guidePos.x + (1.0 - alpha) * targetPos.x,
                         alpha * guidePos.y + (1.0 - alpha) * targetPos.y};
    Point2D toCarrot = {carrotPos.x - robotPose.x, carrotPos.y - robotPose.y};
    double carrotAngle = atan2(toCarrot.y, toCarrot.x);
    double headingAngle = toPInPI(carrotAngle - robotPose.theta);
    vx = hypot(toCarrot.x, toCarrot.y) * cos(headingAngle);
    vy = 0.0;
    vtheta = headingAngle / TURNING_GAIN;
  }

  // Cap the velocities
  vx = std::clamp(vx, -vxLimit, vxLimit);
  vy = std::clamp(vy, -vyLimit, vyLimit);
  vtheta = std::clamp(vtheta, -vthetaLimit, vthetaLimit);

  brain->log->log("tree/Approach",
                  rerun::TextLog(format("vx: %.2f  vy: %.2f  vtheta: %.2f  phase: %d", vx, vy,
                                        vtheta, static_cast<int>(_phase))));
  brain->client->setVelocity(vx, vy, vtheta, false, false, false);
  return NodeStatus::SUCCESS;
}

NodeStatus StrikerDecide::tick() {
  double kickRangeThreshold, kickAngleThreshold;
  getInput("kick_range_threshold", kickRangeThreshold);
  getInput("kick_angle_threshold", kickAngleThreshold);
  string lastDecision, position;
  getInput("decision_in", lastDecision);
  getInput("position", position);

  // ----- Kicker decision logic -----
  double ballGoalDir =
      atan2(-brain->data->ball.posToField.y,
            brain->config->fieldDimensions.length / 2 - brain->data->ball.posToField.x);

  int latestKickerId = -1;
  double latestKickerTime = -1;
  for (const auto &[id, msg] : brain->data->teamMemberMessages) {
    if (msg.kickerElectionTime.seconds() > latestKickerTime) {
      latestKickerId = msg.electedKickerId;
      latestKickerTime = msg.kickerElectionTime.seconds();
    }
  }
  bool isKicker = (latestKickerId == brain->config->playerId);

  // only kicker allowed to decide next kicker after few seconds
  double electionAge = (brain->get_clock()->now() - latestKickerTime).seconds();
  if (isKicker && (electionAge > 4.0)) {
    brain->data->electedKickerId =
        getNewKickerId(brain->config->playerId, ballGoalDir, brain->data);
    brain->data->kickerElectionTime = brain->get_clock()->now();
  }

  // fallback if election is stale (> 6sec)
  if (electionAge > 6.0) {
    brain->data->electedKickerId = brain->config->playerId;  // self is the kicker
    brain->data->kickerElectionTime = brain->get_clock()->now();
    isKicker = true;
  }

  // ----- Kick direction and Descision making logic -----
  double dir_rb_f = brain->data->robotBallAngleToField;
  double ballRange = brain->data->ball.range;
  double robotBallDir = brain->data->robotBallAngleToField;
  auto goalPostAngles = brain->getGoalPostAngles(0.3);

  // TODO: check outfield decision
  bool outField = isOutField(brain->data, brain->config);
  bool shotBlocked = isShotBlocked(ballGoalDir, brain->data->ball.posToField, brain->data);
  bool angleIsGood = (goalPostAngles[0] > dir_rb_f && goalPostAngles[1] < dir_rb_f);
  bool rangeIsGood = (ballRange < kickRangeThreshold);
  bool headingIsGood = (fabs(brain->data->ball.yawToRobot) < kickAngleThreshold);

  string newDecision;
  double kickDir = ballGoalDir;  // Default kick direction
  if (!brain->tree->getEntry<bool>("ball_location_known")) {
    newDecision = "find";
  } else if (outField) {
    newDecision = "reenter";
  } else if (!isKicker) {
    newDecision = "scramble";
    // kickDir = ballGoalDir;  // TODO: place behind the robot
    kickDir = 0;  // place the robot behind the ball
  } else if (shotBlocked && rangeIsGood) {
    // We're the kicker, but can't shoot — fallback to dribbling forward
    newDecision = "dribble";
    kickDir = robotBallDir;
  } else if (!shotBlocked && angleIsGood && rangeIsGood && headingIsGood) {
    newDecision = "kick";
    kickDir = ballGoalDir;
  } else {
    newDecision = "approach";
    kickDir = ballGoalDir;
  }

  setOutput("decision_out", newDecision);
  setOutput("kick_dir_out", kickDir);

  // rerun logging
  static std::map<std::string, uint32_t> decisionColorMap = {
      {"find", 0x0000FFFF},      // Blue
      {"kick", 0xFF0000FF},      // Red
      {"dribble", 0xFFA500FF},   // Orange
      {"approach", 0x00FFFFFF},  // Cyan
      {"scramble", 0x00FF00FF},  // Green
      {"reenter", 0xFFFF00FF}    // Yellow
  };

  auto color = decisionColorMap[newDecision];

  // Draw a line representing the kick direction
  const auto kickLine =
      rerun::LineStrip2D({{brain->data->ball.posToField.x, -brain->data->ball.posToField.y},
                          {brain->data->ball.posToField.x + cos(kickDir) * 1.0,
                           -brain->data->ball.posToField.y - sin(kickDir) * 1.0}});
  brain->log->log("field/kick_dir",
                  rerun::LineStrips2D(kickLine).with_colors({color}).with_radii({0.01}));

  brain->log->logToField(
      "field/StrikerDecide",
      format(
          "Decision: %s | kickDir: %.2f | rbDir: %.2f | ballRange: %.2f | ballYaw: %.2f | "
          "goodAngle: %d | goodRange: %d | goodHeading: %d | isShotBlocked: %d | isLocalKicker: %d",
          newDecision.c_str(), kickDir, dir_rb_f, ballRange, brain->data->ball.yawToRobot,
          angleIsGood, rangeIsGood, headingIsGood, shotBlocked, brain->data->isKicker),
      color, 0.2);
  return NodeStatus::SUCCESS;
}

NodeStatus CheckAndStandUp::tick() {
  if (brain->tree->getEntry<bool>("gc_is_under_penalty") ||
      brain->data->currentRobotModeIndex == 1) {
    brain->data->needManualRelocate = false;
    brain->tree->setEntry<bool>("should_recalibrate_after_fall_recovery", false);
    brain->data->recoveryPerformed = false;
    brain->data->enterDampingPerformed = false;
    brain->log->log("recovery", rerun::TextLog("reset recovery"));
    return NodeStatus::SUCCESS;
  }

  if (brain->data->needManualRelocate) {
    brain->log->log("recovery", rerun::TextLog("need manual relocate"));
    return NodeStatus::FAILURE;
  }

  if (brain->data->recoveryState == RobotRecoveryState::HAS_FALLEN &&
      // brain->data->isRecoveryAvailable && //
      // 倒了就直接尝试RL起身，（不需要关注是否recoveryAailable）
      brain->data->currentRobotModeIndex != 1 &&  // not in prepare
      !brain->data->recoveryPerformed && !brain->data->enterDampingPerformed) {
    brain->client->standUp();
    brain->data->recoveryPerformed = true;
    brain->data->lastRecoveryTime = brain->get_clock()->now();
    brain->log->log("recovery", rerun::TextLog("Fall detect and stand up"));
  }

  // 如果没有起来, 且已经过了 5 秒, 就进入阻尼模式，且只进入一次
  auto now = brain->get_clock()->now();
  auto seconds_elaps = now.seconds() - brain->data->lastRecoveryTime.seconds();
  if (brain->data->recoveryPerformed && !brain->data->enterDampingPerformed && seconds_elaps > 10 &&
      brain->data->recoveryState != RobotRecoveryState::IS_READY) {
    brain->client->enterDamping();
    brain->data->enterDampingPerformed = true;
    brain->tree->setEntry<bool>("should_recalibrate_after_fall_recovery", false);
    brain->log->log("recovery",
                    rerun::TextLog("Enter Damping, seconds_elaps: " + to_string(seconds_elaps) +
                                   "recoveryState: " +
                                   to_string(static_cast<int>(brain->data->recoveryState))));

    // std::cout << "Enter Damping, seconds_elaps: " << seconds_elaps << " recoveryState: " <<
    // static_cast<int>(brain->data->recoveryState) << std::endl;
  }

  if (brain->data->recoveryPerformed && !brain->data->enterDampingPerformed &&
      brain->data->recoveryState == RobotRecoveryState::IS_READY) {
    brain->tree->setEntry<bool>("should_recalibrate_after_fall_recovery", true);
    brain->log->log("recovery",
                    rerun::TextLog("Standup success, seconds_elaps: " + to_string(seconds_elaps) +
                                   "recoveryState: " +
                                   to_string(static_cast<int>(brain->data->recoveryState))));
  }

  // 机器人站着且是robocup步态，可以重置跌到爬起的状态
  if (brain->data->recoveryPerformed &&
      brain->data->recoveryState == RobotRecoveryState::IS_READY &&
      brain->data->currentRobotModeIndex == 8) {  // in robocup gait
    brain->data->recoveryPerformed = false;
    brain->data->enterDampingPerformed = false;
    brain->log->log("recovery",
                    rerun::TextLog("Reset recovery, recoveryState: " +
                                   to_string(static_cast<int>(brain->data->recoveryState))));
  }

  return NodeStatus::SUCCESS;
}

NodeStatus RotateForRelocate::onStart() {
  this->_lastSuccessfulLocalizeTime = brain->data->lastSuccessfulLocalizeTime;
  this->_startTime = brain->get_clock()->now();
  return NodeStatus::RUNNING;
}

NodeStatus RotateForRelocate::onRunning() {
  double vtheta_limit;
  getInput("vtheta_limit", vtheta_limit);
  int max_msec_locate;
  getInput("max_msec_locate", max_msec_locate);

  brain->client->moveHead(0.4, 0.0);
  brain->client->setVelocity(0, 0, vtheta_limit);

  if (this->_lastSuccessfulLocalizeTime.nanoseconds() !=
      brain->data->lastSuccessfulLocalizeTime.nanoseconds()) {
    brain->tree->setEntry<bool>("should_recalibrate_after_fall_recovery", false);
    brain->log->log("recovery", rerun::TextLog("Relocated successfully"));
    brain->client->moveHead(0.0, 0.0);
    brain->client->setVelocity(0, 0, 0);
    return NodeStatus::SUCCESS;
  }

  if (brain->msecsSince(this->_startTime) > max_msec_locate) {
    brain->tree->setEntry<bool>("should_recalibrate_after_fall_recovery", false);
    brain->data->needManualRelocate = true;
    brain->client->enterDamping();
    brain->log->log("recovery", rerun::TextLog("Relocated failed for timeout"));
    return NodeStatus::SUCCESS;
  }

  return NodeStatus::RUNNING;
}

void RotateForRelocate::onHalted() {
  brain->tree->setEntry<bool>("should_recalibrate_after_fall_recovery", false);
}

NodeStatus GoalieDecide::tick() {
  double kickRangeThreshold, kickAngleThreshold;
  getInput("kick_range_threshold", kickRangeThreshold);
  getInput("kick_angle_threshold", kickAngleThreshold);
  string lastDecision, position;
  getInput("decision_in", lastDecision);
  getInput("position", position);

  // ----- Goalie decision logic -----
  bool ballInDangerArea = brain->data->ball.posToField.x < -1.0;
  double velThreshold = (lastDecision == "keepgoal") ? 0.1 : 0.2;
  bool ballStopped = hypot(brain->data->ballVelocityX, brain->data->ballVelocityY) < velThreshold;

  double currTime = brain->get_clock()->now().seconds();
  bool opponentAroundBall = (currTime - brain->data->lastOpponentNearBallTime.seconds()) < 2.0;
  bool ballInDanger = ballInDangerArea && (!ballStopped || opponentAroundBall);
  bool behindBall = (brain->data->robotPoseToField.x - brain->data->ball.posToField.x) < 0.0;

  double ballGoalDir =
      atan2(-brain->data->ball.posToField.y,
            brain->config->fieldDimensions.length / 2 - brain->data->ball.posToField.x);
  bool isKicker = !ballInDanger;
  // bool isKicker = !ballInDanger && isLocalKicker(brain->config->playerId, ballGoalDir,
  // brain->data);
  // for (const auto &[id, msg] : brain->data->teamMemberMessages) {
  //   if (msg.isKicker) {
  //     isKicker = false;
  //     break;
  //   }
  // }
  // brain->data->isKicker = isKicker;

  // ----- Kick direction and Descision making logic -----
  double dir_rb_f = brain->data->robotBallAngleToField;
  double ballRange = brain->data->ball.range;
  double robotBallDir = brain->data->robotBallAngleToField;
  auto goalPostAngles = brain->getGoalPostAngles(0.3);

  bool shotBlocked = isShotBlocked(ballGoalDir, brain->data->ball.posToField, brain->data);
  bool angleIsGood = (goalPostAngles[0] > dir_rb_f && goalPostAngles[1] < dir_rb_f);
  bool rangeIsGood = (ballRange < kickRangeThreshold);
  bool headingIsGood = (fabs(brain->data->ball.yawToRobot) < kickAngleThreshold);

  string newDecision;
  double kickDir = ballGoalDir;  // Default kick direction
  if (!brain->tree->getEntry<bool>("ball_location_known")) {
    newDecision = "find";
  } else if (ballInDanger && behindBall) {
    newDecision = "keepgoal";
  } else if (shotBlocked && rangeIsGood) {
    newDecision = "dribble";
    kickDir = robotBallDir;
  } else if (!shotBlocked && angleIsGood && rangeIsGood && headingIsGood) {
    newDecision = "kick";
    kickDir = ballGoalDir;
  } else {
    newDecision = "approach";
    kickDir = ballGoalDir;
  }

  setOutput("decision_out", newDecision);
  setOutput("kick_dir_out", kickDir);

  // rerun logging
  static std::map<std::string, uint32_t> decisionColorMap = {
      {"find", 0x0000FFFF},      // Blue
      {"kick", 0xFF0000FF},      // Red
      {"dribble", 0xFFA500FF},   // Orange
      {"approach", 0x00FFFFFF},  // White
      {"keepgoal", 0x00FF00FF}   // Green
  };

  auto color = decisionColorMap[newDecision];

  // Draw a line representing the kick direction
  const auto kickLine =
      rerun::LineStrip2D({{brain->data->ball.posToField.x, -brain->data->ball.posToField.y},
                          {brain->data->ball.posToField.x + cos(kickDir) * 1.0,
                           -brain->data->ball.posToField.y - sin(kickDir) * 1.0}});
  brain->log->log("field/kick_dir",
                  rerun::LineStrips2D(kickLine).with_colors({color}).with_radii({0.01}));
  brain->log->logToField(
      "field/GoalieDecide",
      format("Decision: %s | kickDir: %.2f | rbDir: %.2f | ballRange: %.2f | ballYaw: %.2f | "
             "goodAngle: %d | goodRange: %d | goodHeading: %d | isShotBlocked: %d | "
             "ballInDanger: %d | isBallStopped: %d | ballVelX: %.2f | ballVelY: %.2f",
             newDecision.c_str(), kickDir, dir_rb_f, ballRange, brain->data->ball.yawToRobot,
             angleIsGood, rangeIsGood, headingIsGood, shotBlocked, ballInDangerArea, ballStopped,
             brain->data->ballVelocityX, brain->data->ballVelocityY),
      color, 0.2);
  return NodeStatus::SUCCESS;
}

NodeStatus Kick::onStart() {
  _startTime = brain->get_clock()->now();

  double vxLimit, vyLimit;
  getInput("vx_limit", vxLimit);
  getInput("vy_limit", vyLimit);
  // getInput("vtheta_limit", vthetaLimit);
  int minMSecKick;
  getInput("min_msec_kick", minMSecKick);
  double vxFactor = brain->config->vxFactor;
  double yawOffset = brain->config->yawOffset;

  double adjustedYaw = brain->data->ball.yawToRobot - yawOffset;
  double tx = cos(adjustedYaw) * brain->data->ball.range;
  double ty = sin(adjustedYaw) * brain->data->ball.range;

  double vx, vy;

  if (fabs(ty) < 0.01 && fabs(adjustedYaw) < 0.01) {
    vx = vxLimit;
    vy = 0.0;
  } else {
    double dirX = std::cos(adjustedYaw) * vxFactor;
    double dirY = std::sin(adjustedYaw);

    double scaleX = vxLimit / std::max(std::abs(dirX), 1e-6);
    double scaleY = vyLimit / std::max(std::abs(dirY), 1e-6);
    double scale = std::min(scaleX, scaleY);

    vx = dirX * scale;
    vy = dirY * scale;
  }
  double vtheta = 0.0;

  double speed = norm(vx, vy);

  _msecKick = speed > 1e-5 ? minMSecKick + static_cast<int>(brain->data->ball.range / speed * 1000)
                           : minMSecKick;

  brain->log->log("tree/Kick/onStart",
                  rerun::TextLog(format("vx: %.2f  vy: %.2f  vtheta: %.2f  msecKick: %d", vx, vy,
                                        vtheta, _msecKick)));
  brain->client->setVelocity(vx, vy, vtheta, false, false, false);
  return NodeStatus::RUNNING;
}

NodeStatus Kick::onRunning() {
  if (brain->msecsSince(_startTime) < _msecKick) return NodeStatus::RUNNING;

  brain->client->setVelocity(0, 0, 0);
  return NodeStatus::SUCCESS;
}

void Kick::onHalted() {
  _startTime -= rclcpp::Duration(100, 0);
}

NodeStatus RobotFindBall::onStart() {
  if (brain->tree->getEntry<bool>("ball_location_known")) {
    brain->client->setVelocity(0, 0, 0);
    return NodeStatus::SUCCESS;
  }
  _turnDir = brain->data->ball.yawToRobot > 0 ? 1.0 : -1.0;

  return NodeStatus::RUNNING;
}

NodeStatus RobotFindBall::onRunning() {
  if (brain->tree->getEntry<bool>("ball_location_known")) {
    brain->client->setVelocity(0, 0, 0);
    brain->tree->setEntry<bool>("robot_find_ball", false);
    return NodeStatus::SUCCESS;
  }

  double vyawLimit;
  getInput("vtheta_limit", vyawLimit);

  double vx = 0;
  double vy = 0;
  double vtheta = 0;
  brain->client->setVelocity(0, 0, vyawLimit * _turnDir);
  brain->tree->setEntry<bool>("robot_find_ball", true);
  return NodeStatus::RUNNING;
}

void RobotFindBall::onHalted() {
  _turnDir = 1.0;
  brain->tree->setEntry<bool>("robot_find_ball", false);
}

NodeStatus SelfLocate::tick() {
  brain->self_locator->motionUpdate(brain->data->robotPoseToOdom);
  brain->self_locator->sensorUpdate(brain->data->goalposts, brain->data->markings);
  brain->self_locator->resampling();

  brain->data->goalposts.clear();
  brain->data->markings.clear();

  auto pose = brain->self_locator->getPose();
  brain->calibrateOdom(pose.translation.x(), pose.translation.y(), pose.rotation);

  if (brain->self_locator->isGood()) {
    brain->tree->setEntry<bool>("odom_calibrated", true);
    brain->data->lastSuccessfulLocalizeTime = brain->get_clock()->now();
  } else {
    brain->tree->setEntry<bool>("odom_calibrated", false);
  }

  brain->log->log("field/robot_pose_sample",
                  rerun::Points2D({{pose.translation.x(), -pose.translation.y()},
                                   {pose.translation.x() + 0.1 * cos(pose.rotation),
                                    -pose.translation.y() - 0.1 * sin(pose.rotation)}})
                      .with_radii({0.2, 0.1})
                      .with_colors({0x00FFFFFF, 0xFF0000FF}));

  brain->self_locator->logSamples();

  return NodeStatus::SUCCESS;
}

NodeStatus MoveToPoseOnField::tick() {
  double tx, ty, ttheta, longRangeThreshold, turnThreshold, vxLimit, vyLimit, vthetaLimit,
      xTolerance, yTolerance, thetaTolerance;
  getInput("x", tx);
  getInput("y", ty);
  getInput("theta", ttheta);
  getInput("long_range_threshold", longRangeThreshold);
  getInput("turn_threshold", turnThreshold);
  getInput("vx_limit", vxLimit);
  getInput("vx_limit", vxLimit);
  getInput("vy_limit", vyLimit);
  getInput("vtheta_limit", vthetaLimit);
  getInput("x_tolerance", xTolerance);
  getInput("y_tolerance", yTolerance);
  getInput("theta_tolerance", thetaTolerance);
  brain->client->moveToPoseOnField(tx, ty, ttheta, longRangeThreshold, turnThreshold, vxLimit,
                                   vyLimit, vthetaLimit, xTolerance, yTolerance, thetaTolerance);
  return NodeStatus::SUCCESS;
}

NodeStatus KeepGoal::tick() {
  double minBallDist;
  double longRangeThreshold, turnThreshold, vxLimit, vyLimit, vthetaLimit;
  double xTolerance, yTolerance, thetaTolerance;
  getInput("min_ball_dist", minBallDist);
  getInput("long_range_threshold", longRangeThreshold);
  getInput("turn_threshold", turnThreshold);
  getInput("vx_limit", vxLimit);
  getInput("vy_limit", vyLimit);
  getInput("vtheta_limit", vthetaLimit);
  getInput("x_tolerance", xTolerance);
  getInput("y_tolerance", yTolerance);
  getInput("theta_tolerance", thetaTolerance);

  const auto angles = brain->getGoalPostAngles(0.0, true);
  double midAngle = atan2(sin(angles[0]) + sin(angles[1]), cos(angles[0]) + cos(angles[1]));

  double ballGoalDist =
      hypot(brain->data->ball.posToField.x + brain->config->fieldDimensions.length / 2,
            brain->data->ball.posToField.y);
  double goalieDist = std::clamp(brain->data->ball.range, minBallDist, ballGoalDist);

  Point ballPos = brain->data->ball.posToField;
  double tx = ballPos.x + goalieDist * cos(midAngle);
  tx = max(tx, -brain->config->fieldDimensions.length / 2 + 0.5);
  double ty = ballPos.y + goalieDist * sin(midAngle);

  // set the theta to face the ball
  Pose2D robotPose = brain->data->robotPoseToField;
  double ttheta = atan2(ballPos.y - robotPose.y, ballPos.x - robotPose.x);
  brain->client->moveToPoseOnField(tx, ty, ttheta, longRangeThreshold, turnThreshold, vxLimit,
                                   vyLimit, vthetaLimit, xTolerance, yTolerance, thetaTolerance);

  brain->log->log(
      "field/keepgoal",
      rerun::LineStrips2D({
                              rerun::Collection<rerun::Vec2D>{{tx - 0.2, -ty}, {tx + 0.2, -ty}},
                              rerun::Collection<rerun::Vec2D>{{tx, -ty - 0.2}, {tx, -ty + 0.2}},
                          })
          .with_colors({0x00FF00FF})
          .with_radii({0.005})
          .with_draw_order(30));
  return NodeStatus::SUCCESS;
}

NodeStatus BlockGoal::tick() {
  double distToBall;
  double longRangeThreshold, turnThreshold, vxLimit, vyLimit, vthetaLimit;
  double xTolerance, yTolerance, thetaTolerance;
  getInput("dist_to_ball", distToBall);
  getInput("long_range_threshold", longRangeThreshold);
  getInput("turn_threshold", turnThreshold);
  getInput("vx_limit", vxLimit);
  getInput("vy_limit", vyLimit);
  getInput("vtheta_limit", vthetaLimit);
  getInput("x_tolerance", xTolerance);
  getInput("y_tolerance", yTolerance);
  getInput("theta_tolerance", thetaTolerance);

  // striker keep the goal by standing behind the ball
  double ballGoalDir =
      atan2(-brain->data->ball.posToField.y,
            -brain->config->fieldDimensions.length / 2 - brain->data->ball.posToField.x);

  // avoid collision
  double desiredDir = ballGoalDir;
  if (brain->config->playerId == 2) {
    const auto angles = brain->getGoalPostAngles(0.2, true);
    desiredDir = angles[1];
    distToBall += 0.8;
  }

  double tx = brain->data->ball.posToField.x + cos(desiredDir) * distToBall;
  double ty = brain->data->ball.posToField.y + sin(desiredDir) * distToBall;

  // set the theta to face the ball
  Pose2D robotPose = brain->data->robotPoseToField;
  Point ballPos = brain->data->ball.posToField;
  double ttheta = atan2(ballPos.y - robotPose.y, ballPos.x - robotPose.x);
  brain->client->moveToPoseOnField(tx, ty, ttheta, longRangeThreshold, turnThreshold, vxLimit,
                                   vyLimit, vthetaLimit, xTolerance, yTolerance, thetaTolerance);

  brain->log->logToField(
      "field/StrikerDecide",
      format("Decision: BlockGoal | tx: %.2f | ty: %.2f | ttheta: %.2f", tx, ty, ttheta),
      0xFFFFFFFF, 0.2);
  return NodeStatus::SUCCESS;
}

NodeStatus ReEnterField::tick() {
  double margin;
  double longRangeThreshold, turnThreshold, vxLimit, vyLimit, vthetaLimit;
  double xTolerance, yTolerance, thetaTolerance;
  getInput("margin", margin);
  getInput("long_range_threshold", longRangeThreshold);
  getInput("turn_threshold", turnThreshold);
  getInput("vx_limit", vxLimit);
  getInput("vy_limit", vyLimit);
  getInput("vtheta_limit", vthetaLimit);
  getInput("x_tolerance", xTolerance);
  getInput("y_tolerance", yTolerance);
  getInput("theta_tolerance", thetaTolerance);

  const auto &field = brain->config->fieldDimensions;
  Pose2D robotPose = brain->data->robotPoseToField;

  double fieldHalfLength = field.length / 2.0;
  double fieldHalfWidth = field.width / 2.0;

  double dx_left = fabs(robotPose.x + fieldHalfLength);   // our goal
  double dx_right = fabs(robotPose.x - fieldHalfLength);  // opponent goal
  double dy_top = fabs(robotPose.y - fieldHalfWidth);     // top side
  double dy_bottom = fabs(robotPose.y + fieldHalfWidth);  // bottom side

  double min_dist = dx_left;
  double tx = -fieldHalfLength + margin;
  double ty = robotPose.y;
  if (dx_right < min_dist) {
    min_dist = dx_right;
    tx = fieldHalfLength - margin;
    ty = robotPose.y;
  }
  if (dy_top < min_dist) {
    min_dist = dy_top;
    tx = robotPose.x;
    ty = fieldHalfWidth - margin;
  }
  if (dy_bottom < min_dist) {
    min_dist = dy_bottom;
    tx = robotPose.x;
    ty = -fieldHalfWidth + margin;
  }

  // set the theta to face the ball
  Point ballPos = brain->data->ball.posToField;
  double ttheta = atan2(ballPos.y - robotPose.y, ballPos.x - robotPose.x);

  brain->client->moveToPoseOnField(tx, ty, ttheta, longRangeThreshold, turnThreshold, vxLimit,
                                   vyLimit, vthetaLimit, xTolerance, yTolerance, thetaTolerance);

  brain->log->log(
      "field/reenter",
      rerun::LineStrips2D({
                              rerun::Collection<rerun::Vec2D>{{tx - 0.2, -ty}, {tx + 0.2, -ty}},
                              rerun::Collection<rerun::Vec2D>{{tx, -ty - 0.2}, {tx, -ty + 0.2}},
                          })
          .with_colors({0xFFFF00FF})
          .with_radii({0.005})
          .with_draw_order(30));
  return NodeStatus::SUCCESS;
}

NodeStatus WaveHand::tick() {
  string action;
  getInput("action", action);
  if (action == "start")
    brain->client->waveHand(true);
  else
    brain->client->waveHand(false);
  return NodeStatus::SUCCESS;
}

NodeStatus PrintMsg::tick() {
  Expected<std::string> msg = getInput<std::string>("msg");
  if (!msg) {
    throw RuntimeError("missing required input [msg]: ", msg.error());
  }
  std::cout << "[MSG] " << msg.value() << std::endl;
  return NodeStatus::SUCCESS;
}
