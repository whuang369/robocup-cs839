#include "booster_vision/vision_node.h"

#include <functional>
#include <filesystem>
#include <sstream>

#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include "vision_interface/msg/detected_object.hpp"
#include "vision_interface/msg/detections.hpp"

#include "booster_vision/base/data_syncer.hpp"
#include "booster_vision/model/detector.h"
#include "booster_vision/pose_estimator/pose_estimator.h"
#include "booster_vision/base/misc_utils.hpp"
#include "booster_vision/base/pointcloud_process.h"
#include "booster_vision/base/img_bridge.hpp"

#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <Eigen/Dense>  // Requires Eigen library

using std::string;
using std::vector;
using std::placeholders::_1;

namespace {
bool detectPixelShiftCorruption(const cv::Mat &image) {
  if (image.empty()) return false;

  // Convert to grayscale if needed
  cv::Mat gray;
  if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = image.clone();
  }

  // Parameters (can be made configurable)
  const double max_gradient_thresh = 60000.0;
  const double edge_gradient_thresh = 80000.0;
  const int roll_offset = 50;

  // Compute horizontal gradient (vertical edge strength)
  cv::Mat grad, col_strength;
  cv::Sobel(gray, grad, CV_32F, 1, 0, 3);
  cv::reduce(cv::abs(grad), col_strength, 0, cv::REDUCE_SUM, CV_32F);

  // Find the column with maximum strength
  int seam_col = 0;
  double max_strength = 0.0;
  for (int col = 0; col < gray.cols; col++) {
    double strength = col_strength.at<float>(0, col);
    if (strength > max_strength) {
      max_strength = strength;
      seam_col = col;
    }
  }

  // ---- Manual Horizontal Roll ----
  int shift = roll_offset % gray.cols;
  if (shift < 0) shift += gray.cols;

  cv::Mat right = gray.colRange(gray.cols - shift, gray.cols);
  cv::Mat left = gray.colRange(0, gray.cols - shift);
  cv::Mat rolled_image;
  cv::hconcat(right, left, rolled_image);
  // --------------------------------

  // Edge strength after roll
  cv::Mat edge_strength, edge_gradient;
  cv::Sobel(rolled_image, edge_gradient, CV_32F, 1, 0, 3);
  cv::reduce(cv::abs(edge_gradient), edge_strength, 0, cv::REDUCE_SUM, CV_32F);
  double max_edge_strength = 0.0;
  int seam_edge_col = 0;
  for (int col = roll_offset - 5; col < roll_offset + 5; col++) {
    double strength = edge_strength.at<float>(0, col);
    if (strength > max_edge_strength) {
      max_edge_strength = strength;
      seam_edge_col = col;
    }
  }

  bool corrupted = (max_strength > max_gradient_thresh && max_edge_strength < edge_gradient_thresh);
  if (corrupted) {
    std::cout << "[Seam Detected] Column: " << seam_col << ", Max Strength: " << std::fixed
              << std::setprecision(2) << max_strength << ", Edge Strength: " << std::fixed
              << std::setprecision(2) << max_edge_strength << std::endl;
  }
  return corrupted;
}
}  // anonymous namespace

namespace booster_vision {

void VisionNode::Init(const string &cfg_path, const string &cfg_local_path) {
  // load config file
  if (!std::filesystem::exists(cfg_path)) {
    std::cerr << "Error: Configuration file '" << cfg_path << "' does not exist." << std::endl;
    return;
  }

  YAML::Node node = YAML::LoadFile(cfg_path);

  if (std::filesystem::exists(cfg_local_path)) {
    YAML::Node local_node = YAML::LoadFile(cfg_local_path);
    MergeYAML(node, local_node);
    std::cout << "Merged local override config: " << cfg_local_path << std::endl;
  } else {
    std::cout << "No local override config found. Skipping: " << cfg_local_path << std::endl;
  }
  std::cout << "loaded cfg file: " << std::endl << node << std::endl;

  this->declare_parameter<string>("rerunLog.server_addr", "rerun+http://127.0.0.1:9876/proxy");
  this->declare_parameter<bool>("rerunLog.enable", node["rerun_log"].as<bool>());
  string rerun_server_addr;
  bool rerun_enable;
  this->get_parameter("rerunLog.server_addr", rerun_server_addr);
  this->get_parameter("rerunLog.enable", rerun_enable);
  log_ = std::make_shared<VisionLog>(this, rerun_enable, rerun_server_addr,
                                     node["rerun_id"].as<string>(""));

  // read camera param
  if (!node["camera"]) {
    std::cerr << "no camera param found here" << std::endl;
    return;
  } else {
    intr_ = Intrinsics(node["camera"]["intrin"]);
    p_eye2head_ = as_or<Pose>(node["camera"]["extrin"], Pose());

    float pitch_comp = as_or<float>(node["camera"]["pitch_compensation"], 0.0);
    float yaw_comp = as_or<float>(node["camera"]["yaw_compensation"], 0.0);

    p_headprime2head_ = Pose(0, 0, 0, 0, pitch_comp * M_PI / 180, yaw_comp * M_PI / 180);
  }

  // init detector
  if (!node["detection_model"]) {
    std::cerr << "no detection model param here" << std::endl;
    return;
  } else {
    detector_ = YoloV8Detector::CreateYoloV8Detector(node["detection_model"]);
  }

  // init data_syncer
  use_depth_ = as_or<bool>(node["use_depth"], false);
  data_syncer_ = std::make_shared<DataSyncer>(use_depth_);

  // init pose estimator
  pose_estimator_ = std::make_shared<PoseEstimator>(intr_);
  pose_estimator_->Init(YAML::Node());
  pose_estimator_map_["default"] = pose_estimator_;

  if (node["ball_pose_estimator"]) {
    pose_estimator_map_["ball"] = std::make_shared<BallPoseEstimator>(intr_);
    pose_estimator_map_["ball"]->Init(node["ball_pose_estimator"]);
  }

  if (node["human_like_pose_estimator"]) {
    pose_estimator_map_["human_like"] = std::make_shared<HumanLikePoseEstimator>(intr_);
    pose_estimator_map_["human_like"]->Init(node["human_like_pose_estimator"]);
  }

  // init ros related
  string color_topic = "/camera/camera/rgb/image_rect_color";
  string depth_topic = "/camera/camera/depth/depth_registered";
  if (node["topics"]) {
    color_topic = as_or<string>(node["topics"]["color"], color_topic);
    depth_topic = as_or<string>(node["topics"]["depth"], depth_topic);
  }

  it_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
  color_sub_ = it_->subscribe(color_topic, 1, std::bind(&VisionNode::ColorCallback, this, _1));
  depth_sub_ = it_->subscribe(depth_topic, 10, std::bind(&VisionNode::DepthCallback, this, _1));

  pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/head_pose_stamped", 10, std::bind(&VisionNode::PoseCallBack, this, _1));

  detection_pub_ = this->create_publisher<vision_interface::msg::Detections>(
      "/booster_vision/detection", rclcpp::QoS(1));

  //  start processing thread
  processing_thread_ = std::thread([this]() {
    rclcpp::Rate rate(20);  // 20 Hz processing loop

    while (rclcpp::ok()) {
      this->ProcessFrame();
      rate.sleep();
    }
  });
}

void VisionNode::ProcessFrame() {
  // get synced data
  static double last_timestamp = 0.0;
  SyncedDataBlock synced_data = data_syncer_->getLatestSyncedDataBlock();
  if (synced_data.color_data.data.empty()) return;
  if (synced_data.color_data.timestamp <= last_timestamp) return;
  last_timestamp = synced_data.color_data.timestamp;

  cv::Mat color = synced_data.color_data.data;
  cv::Mat depth = synced_data.depth_data.data;
  Pose p_head2base = synced_data.pose_data.data;
  Pose p_eye2base = p_head2base * p_headprime2head_ * p_eye2head_;

  // inference
  auto detections = detector_->Inference(color);

  // pose estimation in field frame
  static const std::unordered_map<string, string> estimator_map = {
      {"Ball", "ball"},
      {"Person", "human_like"},
      {"Opponent", "human_like"},
      {"Goalpost", "human_like"},
  };

  vision_interface::msg::Detections detection_msg;
  detection_msg.header.stamp = rclcpp::Time(synced_data.color_data.timestamp * 1e9);
  detection_msg.header.frame_id = "camera";
  for (auto &detection : detections) {
    detection.class_name = detector_->kClassLabels[detection.class_id];

    // estimator by class name
    std::string group = "default";
    if (auto it = estimator_map.find(detection.class_name); it != estimator_map.end()) {
      group = it->second;
    }
    auto pose_it = pose_estimator_map_.find(group);
    auto pose_estimator = (pose_it != pose_estimator_map_.end() && pose_it->second)
                              ? pose_it->second
                              : pose_estimator_map_.at("default");

    // pose estimation
    Pose pose_by_color = pose_estimator->EstimateByColor(p_eye2base, detection, color);
    Pose pose_by_depth = pose_estimator->EstimateByDepth(p_eye2base, detection, color, depth);

    vision_interface::msg::DetectedObject detection_obj;
    detection_obj.label = detection.class_name;
    detection_obj.confidence = detection.confidence * 100;
    detection_obj.xmin = detection.bbox.x;
    detection_obj.ymin = detection.bbox.y;
    detection_obj.xmax = detection.bbox.x + detection.bbox.width;
    detection_obj.ymax = detection.bbox.y + detection.bbox.height;

    detection_obj.position = pose_by_depth.getTranslationVec();
    detection_obj.position_projection = pose_by_color.getTranslationVec();

    // NOTE: currently, received_pos is not used in brain
    auto xyz = p_head2base.getTranslationVec();
    auto rpy = p_head2base.getEulerAnglesVec();
    detection_obj.received_pos = {xyz[0],
                                  xyz[1],
                                  xyz[2],
                                  static_cast<float>(rpy[0] * 180 / CV_PI),
                                  static_cast<float>(rpy[1] * 180 / CV_PI),
                                  static_cast<float>(rpy[2] * 180 / CV_PI)};

    detection_msg.detected_objects.push_back(detection_obj);
  }

  // publish msg
  detection_pub_->publish(detection_msg);

  log_->logDetections(color, p_eye2base, detection_msg);

  // vector<float> depth_data(depth.begin<float>(), depth.end<float>());
  // log_->log("image/depth",
  //           rerun::DepthImage(depth_data.data(), {static_cast<uint32_t>(depth.cols),
  //                                                 static_cast<uint32_t>(depth.rows)}));
}

void VisionNode::ColorCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
  if (!msg) {
    std::cerr << "empty image message." << std::endl;
    return;
  }

  cv::Mat imgBGR;
  double timestamp = latest_pose_t_.load(std::memory_order_acquire);
  // double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) *
  // 1e-9;

  try {
    imgBGR = toBGRMat(*msg);
  } catch (const std::exception &e) {
    std::cerr << "converting msg to BGR cv::Mat failed: " << e.what() << std::endl;
    return;
  }

  data_syncer_->AddColor(ColorDataBlock(imgBGR, timestamp));
}

void VisionNode::DepthCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
  if (!use_depth_ || !msg) return;

  if (msg->encoding != "32FC1") {
    std::cerr << "Unsupported depth image encoding: " << msg->encoding << std::endl;
    return;
  }

  double timestamp = latest_pose_t_.load(std::memory_order_acquire);
  // double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) *
  // 1e-9;

  // check message size and alignment
  const uint8_t *raw = msg->data.data();
  size_t expected_size = msg->height * msg->width * sizeof(float);
  if (msg->data.size() != expected_size) return;
  if (reinterpret_cast<uintptr_t>(raw) % alignof(float) != 0) return;

  // convert raw data to cv::Mat
  const float *float_ptr = reinterpret_cast<const float *>(raw);
  cv::Mat depth(msg->height, msg->width, CV_32FC1, const_cast<float *>(float_ptr));
  depth = depth.clone();

  // check whether depth image is valid (bottom 25% for field)
  cv::Mat bottom = depth.rowRange(static_cast<int>(depth.rows * 0.75), depth.rows);
  cv::Mat valid_mask = (bottom >= 0.1f) & (bottom <= 20.0f) & (bottom == bottom);
  int valid_pixels = cv::countNonZero(valid_mask);
  float valid_ratio = static_cast<float>(valid_pixels) / (valid_mask.rows * valid_mask.cols);
  if (valid_ratio < 0.7f) return;

  // set invalid value to 0.0
  cv::Mat full_valid_mask = (depth >= 0.0f) & (depth <= 20.0f) & (depth == depth);
  depth.setTo(0.0f, ~full_valid_mask);

  // add depth data to syncer
  data_syncer_->AddDepth(DepthDataBlock(depth, timestamp));
}

void VisionNode::PoseCallBack(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  // NOTE: DO NOT USE MOTION BOARD TIME (two boards are likely to have different time)
  double timestamp = this->get_clock()->now().seconds();

  const auto &p = msg->pose.position;
  const auto &q = msg->pose.orientation;
  Pose pose(p.x, p.y, p.z, q.x, q.y, q.z, q.w);

  data_syncer_->AddPose(PoseDataBlock(pose, timestamp));

  latest_pose_t_.store(timestamp, std::memory_order_release);
}

bool VisionNode::EstimateCameraRollPitch(const cv::Mat &depth_image, float &roll, float &pitch,
                                         float &height) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  // Convert Depth Image (cv::Mat) to PCL Point Cloud
  for (int v = depth_image.rows * 3 / 4; v < depth_image.rows; v += 2) {
    for (int u = 0; u < depth_image.cols; u += 4) {
      float Z = depth_image.at<float>(v, u);         // Get depth value
      if (!std::isnan(Z) && (Z > 0.6) && (Z < 5)) {  // Ignore invalid depth values
        cv::Point2f uv = cv::Point2f(u, v);
        cv::Point3f point2eye = intr_.BackProject(uv, Z);

        // reject points taller than the camera
        if (point2eye.y > 0) {
          cloud->points.emplace_back(point2eye.x, point2eye.y, point2eye.z);
        }
      }
    }
  }

  if (cloud->points.size() < 100) {
    std::cout << "Fewer than 100 points." << std::endl;
    return false;
  }

  // Fit a Plane Using RANSAC
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100);
  seg.setDistanceThreshold(0.05);  // Adjust based on noise level
  seg.setInputCloud(cloud);
  seg.segment(*inliers, *coefficients);

  float confidence = static_cast<float>(inliers->indices.size()) / cloud->size();
  std::cout << "inlier percentage: " << confidence << std::endl;

  if (inliers->indices.empty()) {
    std::cout << "Could not estimate a ground plane." << std::endl;
    return false;
  }

  float A = coefficients->values[0];
  float B = coefficients->values[1];
  float C = coefficients->values[2];
  float D = coefficients->values[3];

  Eigen::Vector3d v(A, B, C);
  v.normalize();
  // Target vector (0,0,1)
  Eigen::Vector3d k(0, 0, 1);

  // Compute rotation axis
  Eigen::Vector3d axis = v.cross(k);
  double sinTheta = axis.norm();
  double cosTheta = v.dot(k);

  // If already aligned, return identity
  if (sinTheta == 0) {
    return false;
  }

  axis.normalize();  // Normalize rotation axis

  // Compute skew-symmetric cross-product matrix
  Eigen::Matrix3d K;
  K << 0, -axis.z(), axis.y(), axis.z(), 0, -axis.x(), -axis.y(), axis.x(), 0;

  // Compute rotation matrix using Rodrigues' formula
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + sinTheta * K + (1 - cosTheta) * K * K;
  double yaw;

  // Extract pitch (θ)
  pitch = std::asin(-R(2, 0));  // R31

  if (std::abs(pitch - M_PI / 2) < 1e-6) {
    // Gimbal lock case: pitch = +90 degrees
    yaw = std::atan2(R(0, 1), R(0, 2));
    roll = 0;  // Arbitrary, since roll is not uniquely defined
  } else if (std::abs(pitch + M_PI / 2) < 1e-6) {
    // Gimbal lock case: pitch = -90 degrees
    yaw = std::atan2(-R(0, 1), -R(0, 2));
    roll = 0;
  } else {
    // Normal case
    yaw = std::atan2(R(1, 0), R(0, 0));   // atan2(R21, R11)
    roll = std::atan2(R(2, 1), R(2, 2));  // atan2(R32, R33)
  }

  std::cout << "Estimated Roll: " << roll * 180.0 / M_PI << " degrees" << std::endl;
  std::cout << "Estimated Pitch: " << pitch * 180.0 / M_PI << " degrees" << std::endl;
  float normal_magnitude = std::sqrt(A * A + B * B + C * C);
  if (normal_magnitude == 0) {
    std::cerr << "Invalid plane coefficients!" << std::endl;
    return false;
  }
  height = std::abs(D) / normal_magnitude;
  std::cout << "Estimated height: " << height << std::endl;

  if (confidence < 0.4) {
    std::cout << "Confidence is too low." << std::endl;
    return false;
  }

  if ((roll * 180.0 / M_PI < 0) && (roll * 180.0 / M_PI > -70) ||
      (roll * 180.0 / M_PI > 0) && (roll * 180.0 / M_PI > 20)) {
    std::cout << "Estimated plane may be a wall." << std::endl;
    return false;
  }
  return true;
}

}  // namespace booster_vision
