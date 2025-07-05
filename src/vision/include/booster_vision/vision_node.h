#pragma once

#include <memory>
#include <map>
#include <string>

#include <opencv2/core.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <image_transport/image_transport.hpp>

#include "vision_interface/msg/detections.hpp"

#include "booster_vision/base/intrin.h"
#include "booster_vision/base/pose.h"
#include "booster_vision/vision_log.h"

namespace booster_vision {

class DataSyncer;
class PoseEstimator;
class YoloV8Detector;

class VisionNode : public rclcpp::Node {
 public:
  VisionNode(const std::string &node_name) : rclcpp::Node(node_name) {}
  ~VisionNode() {
    if (processing_thread_.joinable()) processing_thread_.join();
  }

  void Init(const std::string &cfg_template_path, const std::string &cfg_path);
  void ColorCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
  void DepthCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
  void PoseCallBack(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  bool EstimateCameraRollPitch(const cv::Mat &depth_image, float &roll, float &pitch,
                               float &height);

  void ProcessFrame();

 private:
  std::thread processing_thread_;

  bool use_depth_ = false;

  Intrinsics intr_;
  Pose p_eye2head_;
  Pose p_headprime2head_;
  float z_compensation_ = 0;

  // ROS2 interfaces
  std::shared_ptr<rclcpp::Node> nh_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Publisher<vision_interface::msg::Detections>::SharedPtr detection_pub_;

  std::shared_ptr<image_transport::ImageTransport> it_;
  image_transport::Subscriber color_sub_;
  image_transport::Subscriber depth_sub_;

  // components
  std::shared_ptr<VisionLog> log_;
  std::shared_ptr<DataSyncer> data_syncer_;
  std::shared_ptr<YoloV8Detector> detector_;
  std::shared_ptr<PoseEstimator> pose_estimator_;
  std::map<std::string, std::shared_ptr<PoseEstimator>> pose_estimator_map_;
};

}  // namespace booster_vision
