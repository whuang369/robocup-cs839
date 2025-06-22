#include "brain_config.h"
#include "utils/print.h"
#include "joy_msg.h"

void BrainConfig::handle() {
  // playerStartPos[left, right]
  if (playerStartPos != "left" && playerStartPos != "right") {
    throw invalid_argument("player_start_pos must be one of [left, right]. Got: " + playerStartPos);
  }

  // playerRole [striker, goal_keeper]
  if (playerRole != "striker" && playerRole != "goal_keeper") {
    throw invalid_argument("player_role must be one of [striker, goal_keeper]. Got: " + playerRole);
  }

  // playerId [0, 1, 2, 3]
  if (playerId != 0 && playerId != 1 && playerId != 2 && playerId != 3) {
    throw invalid_argument("[Error] player_id must be one of [0, 1, 2, 3]. Got: " +
                           to_string(playerId));
  }

  // fieldType [adult_size, kid_size]
  if (fieldType == "adult_size") {
    fieldDimensions = FD_ADULTSIZE;
  } else if (fieldType == "kid_size") {
    fieldDimensions = FD_KIDSIZE;
  } else if (fieldType == "gdc") {
    fieldDimensions = FD_GDC;
  } else {
    throw invalid_argument("[Error] fieldType must be one of [adult_size, kid_size, gdc]. Got: " +
                           fieldType);
  }

  // camera
  if (camPixX <= 0 || camPixY <= 0) {
    throw invalid_argument("[Error] camera resolution must be positive. Got: " +
                           to_string(camPixX) + "x" + to_string(camPixY));
  }

  // joystick
  if (joystick == "logitech") {
    JoyMsg::type = JoyMsg::LOGITECH;
  } else if (joystick == "beitong") {
    JoyMsg::type = JoyMsg::BEITONG;
  } else {
    throw invalid_argument("[Error] joystick must be one of [logitech, beitong]. Got: " + joystick);
  }
}

void BrainConfig::print(ostream &os) {
  os << "Configs:" << endl;
  os << "----------------------------------------" << endl;
  os << "teamId = " << teamId << endl;
  os << "playerId = " << playerId << endl;
  os << "fieldType = " << fieldType << endl;
  os << "playerRole = " << playerRole << endl;
  os << "playerStartPos = " << playerStartPos << endl;
  os << "----------------------------------------" << endl;
  os << "robotHeight = " << robotHeight << endl;
  os << "robotOdomFactor = " << robotOdomFactor << endl;
  os << "vxFactor = " << vxFactor << endl;
  os << "yawOffset = " << yawOffset << endl;
  os << "joystick = " << joystick << endl;
  os << "----------------------------------------" << endl;
  os << "imageTopic = " << imageTopic << endl;
  os << "camPixX = " << camPixX << endl;
  os << "camPixY = " << camPixY << endl;
  os << "----------------------------------------" << endl;
  os << "visualOdomTopic = " << visualOdomTopic << endl;
  os << "----------------------------------------" << endl;
  os << "rerunLogEnable = " << rerunLogEnable << endl;
  os << "rerunLogServerAddr = " << rerunLogServerAddr << endl;
  os << "rerunLogImgInterval = " << rerunLogImgInterval << endl;
  os << "----------------------------------------" << endl;
  os << "treeFilePath = " << treeFilePath << endl;
  os << "----------------------------------------" << endl;
}