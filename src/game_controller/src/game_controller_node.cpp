#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

#include "game_controller_node.h"

GameControllerNode::GameControllerNode(string name) : rclcpp::Node(name) {
  _socket = -1;

  // Declare Ros2 parameters. Note that newly added parameters in the configuration file need to be
  // explicitly declared here.
  declare_parameter<int>("port", 3838);
  declare_parameter<bool>("enable_ip_white_list", false);
  declare_parameter<vector<string>>("ip_white_list", vector<string>{});

  // Read parameters from the configuration. Note that the read parameters should be printed in the
  // log for easy problem investigation.
  get_parameter("port", _port);
  RCLCPP_INFO(get_logger(), "[get_parameter] port: %d", _port);
  get_parameter("enable_ip_white_list", _enable_ip_white_list);
  RCLCPP_INFO(get_logger(), "[get_parameter] enable_ip_white_list: %d", _enable_ip_white_list);
  get_parameter("ip_white_list", _ip_white_list);
  RCLCPP_INFO(get_logger(), "[get_parameter] ip_white_list(len=%ld)", _ip_white_list.size());
  for (size_t i = 0; i < _ip_white_list.size(); i++) {
    RCLCPP_INFO(get_logger(), "[get_parameter]     --[%ld]: %s", i, _ip_white_list[i].c_str());
  }

  // Create a publisher and publish to /robocup/game_controller
  _publisher = create_publisher<game_controller_interface::msg::GameControlData>(
      "/robocup/game_controller", 10);
}

GameControllerNode::~GameControllerNode() {
  if (_socket >= 0) {
    close(_socket);
  }

  if (_thread.joinable()) {
    _thread.join();
  }
}

/**
 * Create a Socket and bind it to the specified port.
 */
void GameControllerNode::init() {
  _socket = socket(AF_INET, SOCK_DGRAM, 0);
  if (_socket < 0) {
    RCLCPP_ERROR(get_logger(), "socket failed: %s", strerror(errno));
    throw runtime_error(strerror(errno));
  }

  sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(_port);

  if (bind(_socket, (sockaddr *)&addr, sizeof(addr)) < 0) {
    RCLCPP_ERROR(get_logger(), "bind failed: %s (port=%d)", strerror(errno), _port);
    throw runtime_error(strerror(errno));
  }

  RCLCPP_INFO(get_logger(), "Listening for UDP broadcast on 0.0.0.0:%d", _port);

  // Start a new thread to receive data. The main thread enters the Node's own spin to handle some
  // services of the Node itself.
  _thread = thread(&GameControllerNode::spin, this);
}

void GameControllerNode::spin() {
  // Used to obtain the remote address.
  sockaddr_in remote_addr;
  socklen_t remote_addr_len = sizeof(remote_addr);

  // The 'data' and'msg' are reused within the loop. Pay attention to this point when updating the
  // code in the future.
  RoboCupGameControlData data;
  game_controller_interface::msg::GameControlData msg;

  while (rclcpp::ok()) {
    // Receive data packets from the socket. Expect to receive complete data packets.
    ssize_t ret =
        recvfrom(_socket, &data, sizeof(data), 0, (sockaddr *)&remote_addr, &remote_addr_len);
    if (ret < 0) {
      RCLCPP_ERROR(get_logger(), "receiving UDP message failed: %s", strerror(errno));
      continue;
    }

    // Obtain the remote address
    string remote_ip = inet_ntoa(remote_addr.sin_addr);

    // Ignore incomplete packets
    if (ret != sizeof(data)) {
      RCLCPP_INFO(get_logger(), "packet from %s invalid length=%ld", remote_ip.c_str(), ret);
      continue;
    }

    if (data.version != GAMECONTROLLER_STRUCT_VERSION) {
      RCLCPP_INFO(get_logger(), "packet from %s invalid version: %d", remote_ip.c_str(),
                  data.version);
      continue;
    }

    // filter
    if (!check_ip_white_list(remote_ip)) {
      RCLCPP_INFO(get_logger(), "received packet from %s, but not in ip white list, ignore it",
                  remote_ip.c_str());
      continue;
    }

    // handle packet
    handle_packet(data, msg);

    // publish
    _publisher->publish(msg);

    RCLCPP_INFO(get_logger(), "handle packet successfully ip=%s, packet_number=%d",
                remote_ip.c_str(), data.packetNumber);
  }
}

/**
 * Check whether the IP is in the whitelist. Return true if the whitelist is not enabled or the IP
 * is in the whitelist, and return false in other cases.
 */
bool GameControllerNode::check_ip_white_list(string ip) {
  // Return true if it is not enabled or is in the whitelist.
  if (!_enable_ip_white_list) {
    return true;
  }
  for (size_t i = 0; i < _ip_white_list.size(); i++) {
    if (ip == _ip_white_list[i]) {
      return true;
    }
  }
  return false;
}

/**
 * Convert the UDP data format to the custom Ros2 message format (copy field by field).
 * If any changes are needed, be sure to carefully check each field.
 */
void GameControllerNode::handle_packet(const RoboCupGameControlData &data,
                                       game_controller_interface::msg::GameControlData &msg) {
  // The length of the header is fixed at 4.
  for (int i = 0; i < 4; i++) {
    msg.header[i] = data.header[i];
  }
  msg.version = data.version;
  msg.packet_number = data.packetNumber;
  msg.players_per_team = data.playersPerTeam;
  msg.game_type = data.gameType;
  msg.state = data.state;
  msg.first_half = data.firstHalf;
  msg.kick_off_team = data.kickOffTeam;
  msg.secondary_state = data.secondaryState;
  // The length of secondary_state_info is fixed at 4.
  for (int i = 0; i < 4; i++) {
    msg.secondary_state_info[i] = data.secondaryStateInfo[i];
  }
  msg.drop_in_team = data.dropInTeam;
  msg.drop_in_time = data.dropInTime;
  msg.secs_remaining = data.secsRemaining;
  msg.secondary_time = data.secondaryTime;

  /// The length of teams is fixed at 2.
  for (int i = 0; i < 2; i++) {
    const TeamInfo &team_in = data.teams[i];
    auto &team_out = msg.teams[i];

    // --- Team Info ---
    team_out.team_number = team_in.teamNumber;
    team_out.team_colour = team_in.teamColour;
    team_out.score = team_in.score;
    team_out.penalty_shot = team_in.penaltyShot;
    team_out.single_shots = team_in.singleShots;
    team_out.coach_sequence = team_in.coachSequence;

    // --- Coach Message ---
    team_out.coach_message.clear();
    team_out.coach_message.insert(team_out.coach_message.end(), std::begin(team_in.coachMessage),
                                  std::end(team_in.coachMessage));

    // --- Coach ---
    team_out.coach.penalty = team_in.coach.penalty;
    team_out.coach.secs_till_unpenalised = team_in.coach.secsTillUnpenalised;
    team_out.coach.number_of_warnings = team_in.coach.numberOfWarnings;
    team_out.coach.yellow_card_count = team_in.coach.yellowCardCount;
    team_out.coach.red_card_count = team_in.coach.redCardCount;
    team_out.coach.goal_keeper = team_in.coach.goalKeeper;

    // --- Players ---
    int players_len = sizeof(data.teams[i].players) / sizeof(data.teams[i].players[0]);
    RCLCPP_INFO(get_logger(), "team[%d] players_len=%d", i, players_len);
    team_out.players.clear();
    for (int j = 0; j < players_len; ++j) {
      const RobotInfo &player_in = team_in.players[j];
      game_controller_interface::msg::RobotInfo player_out;

      player_out.penalty = player_in.penalty;
      player_out.secs_till_unpenalised = player_in.secsTillUnpenalised;
      player_out.number_of_warnings = player_in.numberOfWarnings;
      player_out.yellow_card_count = player_in.yellowCardCount;
      player_out.red_card_count = player_in.redCardCount;
      player_out.goal_keeper = player_in.goalKeeper;

      team_out.players.push_back(std::move(player_out));
    }

    for (int j = 0; j < 3 && j < static_cast<int>(team_out.players.size()); ++j) {
      RCLCPP_INFO(get_logger(),
                  "team[%d] player[%d] penalty=%d, secs_till_unpenalised=%d, "
                  "number_of_warnings=%d, yellow_card_count=%d, red_card_count=%d, "
                  "goal_keeper=%d",
                  i, j, team_out.players[j].penalty, team_out.players[j].secs_till_unpenalised,
                  team_out.players[j].number_of_warnings, team_out.players[j].yellow_card_count,
                  team_out.players[j].red_card_count, team_out.players[j].goal_keeper);
    }
  }
}