#include "booster_vision/pose_estimator/pose_estimator.h"

#include "booster_vision/base/misc_utils.hpp"
#include "booster_vision/base/pointcloud_process.h"

namespace booster_vision {

cv::Point3f CalculatePositionByIntersection(const Pose &p_eye2base, const cv::Point2f target_uv,
                                            const Intrinsics &intr) {
  cv::Point3f normalized_point3d = intr.BackProject(target_uv);

  cv::Mat mat_obj_ray =
      (cv::Mat_<float>(3, 1) << normalized_point3d.x, normalized_point3d.y, normalized_point3d.z);
  cv::Mat mat_rot = p_eye2base.getRotationMatrix();
  cv::Mat mat_trans = p_eye2base.toCVMat().col(3).rowRange(0, 3);

  cv::Mat mat_rot_obj_ray = mat_rot * mat_obj_ray;

  float scale = -mat_trans.at<float>(2, 0) / mat_rot_obj_ray.at<float>(2, 0);

  cv::Mat mat_position = mat_trans + scale * mat_rot_obj_ray;
  return cv::Point3f(mat_position.at<float>(0, 0), mat_position.at<float>(1, 0),
                     mat_position.at<float>(2, 0));
}

Pose PoseEstimator::EstimateByColor(const Pose &p_eye2base, const DetectionRes &detection,
                                    const cv::Mat &rgb) {
  // TODO(GW): add modification for cross class
  auto bbox = detection.bbox;
  cv::Point2f target_uv = cv::Point2f(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
  cv::Point3f target_xyz = CalculatePositionByIntersection(p_eye2base, target_uv, intr_);
  return Pose(target_xyz.x, target_xyz.y, target_xyz.z, 0, 0, 0);
}

Pose PoseEstimator::EstimateByDepth(const Pose &p_eye2base, const DetectionRes &detection,
                                    const cv::Mat &depth) {
  const auto &bbox = detection.bbox;
  if (depth.empty() || bbox.width <= 1 || bbox.height <= 1) return Pose();

  int shrink = 0.25f * std::min(bbox.width, bbox.height);
  cv::Rect roi(bbox.x + shrink, bbox.y + shrink, bbox.width - 2 * shrink, bbox.height - 2 * shrink);
  roi &= cv::Rect(0, 0, depth.cols, depth.rows);
  if (roi.width <= 1 || roi.height <= 1) return Pose();

  std::vector<float> valid_depths;
  for (int y = roi.y; y < roi.y + roi.height; ++y) {
    const float *row = depth.ptr<float>(y);
    for (int x = roi.x; x < roi.x + roi.width; ++x) {
      float z = row[x];
      if (z > 0.1f && z < 20.0f && std::isfinite(z)) {
        valid_depths.push_back(z);
      }
    }
  }
  if (valid_depths.size() < 10) return Pose();  // not enough valid depth data

  // calculate average depth (trimmed mean)
  std::sort(valid_depths.begin(), valid_depths.end());
  size_t trim = valid_depths.size() / 10;  // 10% trim
  size_t start = trim;
  size_t end = valid_depths.size() - trim;
  if (end <= start) return Pose();

  float sum = 0.0f;
  for (size_t i = start; i < end; ++i) {
    sum += valid_depths[i];
  }
  float avg_depth = sum / (end - start);

  cv::Point2f uv = cv::Point2f(bbox.x + bbox.width * 0.5f, bbox.y + bbox.height * 0.5f);
  cv::Point3f point = intr_.BackProject(uv, avg_depth);

  // transform point from eye to base frame
  Pose p_obj2eye = Pose(point.x, point.y, point.z, 0, 0, 0);
  Pose p_obj2base = p_eye2base * p_obj2eye;
  return p_obj2base;
}

void BallPoseEstimator::Init(const YAML::Node &node) {
  use_depth_ = as_or<bool>(node["use_depth"], false);
  radius_ = as_or<float>(node["radius"], 0.109);
  downsample_leaf_size_ = as_or<float>(node["downsample_leaf_size"], 0.01);
  cluster_distance_threshold_ = as_or<float>(node["cluster_distance_threshold"], 0.01);
  fitting_distance_threshold_ = as_or<float>(node["fitting_distance_threshold"], 0.01);
}

Pose BallPoseEstimator::EstimateByColor(const Pose &p_eye2base, const DetectionRes &detection,
                                        const cv::Mat &rgb) {
  auto bbox = detection.bbox;
  cv::Point2f target_uv = cv::Point2f(bbox.x + bbox.width / 2, bbox.y + bbox.height);
  cv::Point3f target_xyz = CalculatePositionByIntersection(p_eye2base, target_uv, intr_);
  return Pose(target_xyz.x, target_xyz.y, target_xyz.z, 0, 0, 0);
}

Pose BallPoseEstimator::EstimateByDepth(const Pose &p_eye2base, const DetectionRes &detection,
                                        const cv::Mat &depth) {
  const auto &bbox = detection.bbox;
  if (depth.empty() || bbox.width <= 1 || bbox.height <= 1) return Pose();

  int shrink = 0.25f * std::min(bbox.width, bbox.height);
  cv::Rect roi(bbox.x + shrink, bbox.y + shrink, bbox.width - 2 * shrink, bbox.height - 2 * shrink);
  roi &= cv::Rect(0, 0, depth.cols, depth.rows);
  if (roi.width <= 1 || roi.height <= 1) return Pose();

  std::vector<float> valid_depths;
  for (int y = roi.y; y < roi.y + roi.height; ++y) {
    const float *row = depth.ptr<float>(y);
    for (int x = roi.x; x < roi.x + roi.width; ++x) {
      float z = row[x];
      if (z > 0.1f && z < 20.0f && std::isfinite(z)) {
        valid_depths.push_back(z);
      }
    }
  }
  if (valid_depths.size() < 10) return Pose();  // not enough valid depth data

  // calculate average depth (trimmed mean)
  std::sort(valid_depths.begin(), valid_depths.end());
  size_t trim = valid_depths.size() / 10;  // 10% trim
  size_t start = trim;
  size_t end = valid_depths.size() - trim;
  if (end <= start) return Pose();

  float sum = 0.0f;
  for (size_t i = start; i < end; ++i) {
    sum += valid_depths[i];
  }
  float avg_depth = sum / (end - start);

  cv::Point2f uv = cv::Point2f(bbox.x + bbox.width * 0.5f, bbox.y + bbox.height * 0.5f);
  cv::Point3f point = intr_.BackProject(uv, avg_depth + radius_);

  Pose p_obj2eye = Pose(point.x, point.y, point.z, 0, 0, 0);
  Pose p_obj2base = p_eye2base * p_obj2eye;
  return p_obj2base;
}

void HumanLikePoseEstimator::Init(const YAML::Node &node) {
  use_depth_ = as_or<bool>(node["use_depth"], false);
  downsample_leaf_size_ = as_or<float>(node["downsample_leaf_size"], 0.01);
  statistic_outlier_multiplier_ = as_or<float>(node["statistic_outlier_multiplier"], 0.01);
  fitting_distance_threshold_ = as_or<float>(node["fitting_distance_threshold"], 0.01);
}

Pose HumanLikePoseEstimator::EstimateByColor(const Pose &p_eye2base, const DetectionRes &detection,
                                             const cv::Mat &rgb) {
  auto bbox = detection.bbox;
  cv::Point2f target_uv = cv::Point2f(bbox.x + bbox.width / 2, bbox.y + bbox.height);
  cv::Point3f target_xyz = CalculatePositionByIntersection(p_eye2base, target_uv, intr_);
  return Pose(target_xyz.x, target_xyz.y, target_xyz.z, 0, 0, 0);
}

Pose HumanLikePoseEstimator::EstimateByDepth(const Pose &p_eye2base, const DetectionRes &detection,
                                             const cv::Mat &depth) {
  auto bbox = detection.bbox;
  const int margin = 2;
  const int strip_height = std::max(1, bbox.height / 10);
  cv::Rect roi(bbox.x + margin, bbox.y + bbox.height - strip_height, bbox.width - 2 * margin,
               strip_height);
  roi &= cv::Rect(0, 0, depth.cols, depth.rows);
  if (roi.width <= 1 || roi.height <= 1) return Pose();  // invalid bbox

  std::vector<float> valid_depths;
  for (int y = roi.y; y < roi.y + roi.height; ++y) {
    const float *row = depth.ptr<float>(y);
    for (int x = roi.x; x < roi.x + roi.width; ++x) {
      float z = row[x];
      if (z > 0.1f && z < 20.0f && std::isfinite(z)) {
        valid_depths.push_back(z);
      }
    }
  }
  if (valid_depths.size() < 10) return Pose();  // not enough valid depth data

  // calculate average depth (trimmed mean)
  std::sort(valid_depths.begin(), valid_depths.end());
  size_t trim = valid_depths.size() / 10;  // 10% trim
  size_t start = trim;
  size_t end = valid_depths.size() - trim;
  if (end <= start) return Pose();

  float sum = 0.0f;
  for (size_t i = start; i < end; ++i) {
    sum += valid_depths[i];
  }
  float avg_depth = sum / (end - start);

  cv::Point2f uv = cv::Point2f(bbox.x + bbox.width * 0.5f, bbox.y + bbox.height);
  cv::Point3f point = intr_.BackProject(uv, avg_depth);

  // transform point from eye to base frame
  Pose p_obj2eye = Pose(point.x, point.y, point.z, 0, 0, 0);
  Pose p_obj2base = p_eye2base * p_obj2eye;
  return p_obj2base;
}

}  // namespace booster_vision
