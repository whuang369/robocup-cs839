cmake_minimum_required(VERSION 3.8)
project(brain)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(rerun_sdk REQUIRED)
find_package(backward_ros REQUIRED)

find_package(booster_interface REQUIRED)
find_package(booster_msgs REQUIRED)
find_package(game_controller_interface REQUIRED)
find_package(vision_interface REQUIRED)
find_package(bhuman REQUIRED)

# source files
set(SOURCE_FILES
  src/main.cpp
  src/brain.cpp
  src/brain_tree.cpp
  src/brain_config.cpp
  src/brain_data.cpp
  src/brain_log.cpp
  src/brain_communication.cpp
  src/joy_msg.cpp
  src/locator.cpp
  src/robot_client.cpp
  src/self_localization.cpp
)
add_executable(brain_node ${SOURCE_FILES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -Wno-unused-parameter -Wno-sign-compare -Wno-missing-field-initializers -Wno-unused-variable")

target_link_libraries(brain_node rerun_sdk)
ament_target_dependencies(brain_node rclcpp std_msgs sensor_msgs geometry_msgs
  OpenCV EIGEN3 behaviortree_cpp
  booster_msgs booster_interface game_controller_interface vision_interface bhuman)

target_include_directories(brain_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(TARGETS
  brain_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY behavior_trees
  DESTINATION share/${PROJECT_NAME})

ament_package()
