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
  const int expected_shift = 515;
  const int window = 5;
  const double gradient_thresh = 60000.0;
  const double jump_thresh = 6.0;

  // Compute horizontal gradient (vertical edge strength)
  cv::Mat grad, col_strength;
  cv::Sobel(gray, grad, CV_32F, 1, 0, 3);
  cv::reduce(cv::abs(grad), col_strength, 0, cv::REDUCE_SUM, CV_32F);

  // Search for strong peak around expected shift
  int start = std::max(0, expected_shift - window);
  int end = std::min(gray.cols - 2, expected_shift + window);

  // Find the column with maximum strength in the window
  int seam_col = start;
  double max_strength = 0.0;
  for (int col = start; col < end; col++) {
    double strength = col_strength.at<float>(0, col);
    if (strength > max_strength) {
      max_strength = strength;
      seam_col = col;
    }
  }

  // Compare adjacent columns at seam
  cv::Mat col_left, col_right, diff;
  gray.col(seam_col).convertTo(col_left, CV_32F);
  gray.col(seam_col + 1).convertTo(col_right, CV_32F);
  cv::absdiff(col_left, col_right, diff);
  double mean_jump = cv::mean(diff)[0];

  bool corrupted = (max_strength > gradient_thresh && mean_jump > jump_thresh);
  if (corrupted) {
    std::cout << "[Seam Detected] Column: " << seam_col << ", Edge Strength: " << std::fixed
              << std::setprecision(2) << max_strength << ", Intensity Jump: " << std::fixed
              << std::setprecision(2) << mean_jump << std::endl;
  }
  return corrupted;
}
}  // anonymous namespace

namespace booster_vision {

void VisionNode::Init(const std::string &cfg_path, const std::string &cfg_local_path) {
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

  this->declare_parameter<string>("rerunLog.server_addr", "");
  string rerun_server_addr;
  this->get_parameter("rerunLog.server_addr", rerun_server_addr);
  log = std::make_shared<VisionLog>(this, node["rerun_log"].as<bool>(), rerun_server_addr);

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
  color_sub_ = it_->subscribe(color_topic, 1,
                              std::bind(&VisionNode::ColorCallback, this, std::placeholders::_1));
  depth_sub_ = it_->subscribe(depth_topic, 1,
                              std::bind(&VisionNode::DepthCallback, this, std::placeholders::_1));

  pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
      "/head_pose", 10, std::bind(&VisionNode::PoseCallBack, this, std::placeholders::_1));

  detection_pub_ = this->create_publisher<vision_interface::msg::Detections>(
      "/booster_vision/detection", rclcpp::QoS(1));
}

void VisionNode::ColorCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
  if (!msg) {
    std::cerr << "empty image message." << std::endl;
    return;
  }

  cv::Mat imgBGR;
  try {
    imgBGR = toBGRMat(*msg);
  } catch (const std::exception &e) {
    std::cerr << "converting msg to BGR cv::Mat failed: " << e.what() << std::endl;
    return;
  }

  double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
  vision_interface::msg::Detections detection_msg;
  detection_msg.header = msg->header;

  bool is_corrupted = detectPixelShiftCorruption(imgBGR);
  detection_msg.corrupted_frame = is_corrupted;

  if (is_corrupted) {
    std::cerr << "Pixel shift corruption detected, dropping frame." << std::endl;
    detection_pub_->publish(detection_msg);

    // TODO: remove this
    vector<uint8_t> compressed_image;
    cv::imencode(".jpg", imgBGR, compressed_image, {cv::IMWRITE_JPEG_QUALITY, 10});
    log->setTimeSeconds(timestamp);
    log->log("image/corrupted", rerun::EncodedImage::from_bytes(compressed_image));
    return;
  }

  // get synced data
  SyncedDataBlock synced_data = data_syncer_->getSyncedDataBlock(ColorDataBlock(imgBGR, timestamp));
  cv::Mat color = synced_data.color_data.data;
  cv::Mat depth = synced_data.depth_data.data;
  Pose p_head2base = synced_data.pose_data.data;
  Pose p_eye2base = p_head2base * p_headprime2head_ * p_eye2head_;
  // std::cout << "p_head2base: \n" << p_head2base.toCVMat() << std::endl;
  // std::cout << "p_eye2base: \n" << p_eye2base.toCVMat() << std::endl;
  std::vector<float> eye2base_angles = p_eye2base.getEulerAnglesVec();
  std::vector<float> eye2base_t = p_eye2base.getTranslationVec();
  // std::cout << "p_eye2base roll: " << eye2base_angles[0] * 180.0 / M_PI << std::endl;
  // std::cout << "p_eye2base pitch: " << eye2base_angles[1] * 180.0 / M_PI << std::endl;
  // std::cout << "p_eye2base yaw: " << eye2base_angles[2] * 180.0 / M_PI << std::endl;

  bool estimateCameraRollPitch = false;
  Pose p_eye2base_from_depth;
  if (estimateCameraRollPitch) {
    float roll = 0.0, pitch = 0.0, height = 0.0;
    bool success = EstimateCameraRollPitch(depth, roll, pitch, height);
    if (success) {
      p_eye2base_from_depth =
          Pose(eye2base_t[0], eye2base_t[1], eye2base_t[2], roll, pitch, eye2base_angles[2]);
      std::cout << "p_eye2base_from_depth: \n" << p_eye2base_from_depth.toCVMat() << std::endl;
      // p_eye2base = p_eye2base_from_depth;
    } else
      estimateCameraRollPitch = false;
  }

  // inference
  auto detections = detector_->Inference(color);
  std::cout << detections.size() << " objects detected." << std::endl;

  auto get_estimator = [&](const std::string &class_name) {
    if (class_name == "Ball") {
      return pose_estimator_map_.find("ball") != pose_estimator_map_.end()
                 ? pose_estimator_map_["ball"]
                 : pose_estimator_map_["default"];
    } else if (class_name == "Person" || class_name == "Opponent" || class_name == "Goalpost") {
      return pose_estimator_map_.find("human_like") != pose_estimator_map_.end()
                 ? pose_estimator_map_["human_like"]
                 : pose_estimator_map_["default"];
    } else if (class_name.find("Cross") != std::string::npos || class_name == "PenaltyPoint") {
      return pose_estimator_map_.find("field_marker") != pose_estimator_map_.end()
                 ? pose_estimator_map_["field_marker"]
                 : pose_estimator_map_["default"];
    } else {
      return pose_estimator_map_["default"];
    }
  };

  for (auto &detection : detections) {
    vision_interface::msg::DetectedObject detection_obj;

    detection.class_name = detector_->kClassLabels[detection.class_id];

    auto pose_estimator = get_estimator(detection.class_name);
    Pose pose_obj_by_color = pose_estimator->EstimateByColor(p_eye2base, detection, color);

    // Pose pose_obj_by_depth = pose_estimator->EstimateByDepth(p_eye2base, detection, depth);
    /*
if (estimateCameraRollPitch) {
Pose pose_obj_by_color_est = pose_estimator->EstimateByColor(p_eye2base_from_depth, detection,
color); Pose pose_obj_by_depth_est = pose_estimator->EstimateByDepth(p_eye2base_from_depth,
detection, depth); if (detection.class_name == "XCross" || detection.class_name == "PenaltyPoint") {
    std::cout << detection.class_name << " kinematics color: " <<
pose_obj_by_color.getTranslation()[0] << ", " << pose_obj_by_color.getTranslation()[1] << std::endl;
    std::cout << detection.class_name << " kinematics depth: " <<
pose_obj_by_depth.getTranslation()[0] << ", " << pose_obj_by_depth.getTranslation()[1] << std::endl;
    std::cout << detection.class_name << " estimated color: " <<
pose_obj_by_color_est.getTranslation()[0] << ", " << pose_obj_by_color_est.getTranslation()[1] <<
std::endl; std::cout << detection.class_name << " estimated depth: " <<
pose_obj_by_depth_est.getTranslation()[0] << ", " << pose_obj_by_depth_est.getTranslation()[1] <<
std::endl;
}
}
    */

    detection_obj.position_projection = pose_obj_by_color.getTranslationVec();
    // detection_obj.position = pose_obj_by_depth.getTranslationVec();

    auto xyz = p_head2base.getTranslationVec();
    auto rpy = p_head2base.getEulerAnglesVec();
    std::vector<float> xyzrpy = {xyz[0],
                                 xyz[1],
                                 xyz[2],
                                 static_cast<float>(rpy[0] / CV_PI * 180),
                                 static_cast<float>(rpy[1] / CV_PI * 180),
                                 static_cast<float>(rpy[2] / CV_PI * 180)};
    detection_obj.received_pos = xyzrpy;
    // NOTE: currently, received_pos is not used in brain

    detection_obj.confidence = detection.confidence * 100;
    detection_obj.xmin = detection.bbox.x;
    detection_obj.ymin = detection.bbox.y;
    detection_obj.xmax = detection.bbox.x + detection.bbox.width;
    detection_obj.ymax = detection.bbox.y + detection.bbox.height;
    detection_obj.label = detection.class_name;
    detection_msg.detected_objects.push_back(detection_obj);
  }

  // publish msg
  detection_pub_->publish(detection_msg);

  // rerun logging
  vector<uint8_t> compressed_image;
  cv::imencode(".jpg", color, compressed_image, {cv::IMWRITE_JPEG_QUALITY, 10});

  log->setTimeSeconds(timestamp);
  log->log("image/color", rerun::EncodedImage::from_bytes(compressed_image));

  static std::map<std::string, rerun::Color> detectColorMap = {
      {"Ball", rerun::Color(0xFFFFFFFF)},          // White
      {"LCross", rerun::Color(0xFFFF00FF)},        // Yellow
      {"TCross", rerun::Color(0x00FF00FF)},        // Bright Green
      {"XCross", rerun::Color(0x00FFFFFF)},        // Cyan / Aqua
      {"Person", rerun::Color(0xFF69B4FF)},        // Hot Pink
      {"Goalpost", rerun::Color(0xFFA500FF)},      // Orange
      {"Opponent", rerun::Color(0xFF4500FF)},      // Orange-Red
      {"Corruption", rerun::Color(0xDC143CFF)},    // Crimson (bright red)
      {"PenaltyPoint", rerun::Color(0x7C00FFFF)},  // Vivid Purple/Violet
  };

  vector<rerun::Vec2D> mins, sizes;
  vector<rerun::Text> labels;
  vector<rerun::Color> colors;
  mins.reserve(detections.size());
  sizes.reserve(detections.size());
  labels.reserve(detections.size());
  colors.reserve(detections.size());

  for (const auto &detection : detections) {
    std::ostringstream oss;
    // clang-format off
    oss << detection.class_name << " "
        << "c:" << std::fixed << std::setprecision(2) << detection.confidence;
    // TODO: add position info
    // clang-format on
    labels.emplace_back(rerun::Text(oss.str()));
    mins.emplace_back(rerun::Vec2D(detection.bbox.x, detection.bbox.y));
    sizes.emplace_back(rerun::Vec2D(detection.bbox.width, detection.bbox.height));
    colors.emplace_back(detectColorMap[detection.class_name]);
  }
  log->log(
      "image/detection_boxes",
      rerun::Boxes2D::from_mins_and_sizes(mins, sizes).with_labels(labels).with_colors(colors));
}

void VisionNode::DepthCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
  if (!msg) {
    std::cerr << "empty depth message." << std::endl;
    return;
  }

  cv::Mat depth;
  if (msg->encoding == "16UC1") {
    cv::Mat depthRaw(msg->height, msg->width, CV_16UC1, const_cast<uchar *>(msg->data.data()));
    depthRaw.convertTo(depth, CV_32FC1, 0.001);  // mm to meters
  } else if (msg->encoding == "32FC1") {
    depth = cv::Mat(msg->height, msg->width, CV_32FC1, const_cast<uchar *>(msg->data.data()));
  } else {
    std::cerr << "Unsupported depth image encoding: " << msg->encoding << std::endl;
    return;
  }

  if (depth.empty()) {
    std::cerr << "empty image recevied." << std::endl;
    return;
  }

  double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
  data_syncer_->AddDepth(DepthDataBlock(depth, timestamp));
}

void VisionNode::PoseCallBack(const geometry_msgs::msg::Pose::SharedPtr msg) {
  auto current_time = this->get_clock()->now();
  double timestamp = static_cast<double>(current_time.nanoseconds()) * 1e-9;

  float x = msg->position.x;
  float y = msg->position.y;
  float z = msg->position.z;
  float qx = msg->orientation.x;
  float qy = msg->orientation.y;
  float qz = msg->orientation.z;
  float qw = msg->orientation.w;
  auto pose = Pose(x, y, z, qx, qy, qz, qw);
  data_syncer_->AddPose(PoseDataBlock(pose, timestamp));
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
