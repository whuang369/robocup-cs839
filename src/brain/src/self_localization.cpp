/**
 * @file UKFRobotPoseHypothesis.cpp
 *
 * Implementation of a robot pose estimate based on an Unscented Kalman Filter
 *
 * @author Tim Laue
 * @author Colin Graf
 * @author Yuqian Jiang
 */

#include "self_localization.h"
#include "Math/Covariance.h"
#include "Math/Geometry.h"
#include "Math/Probabilistics.h"
#include "Math/Eigen.h"
#include "utils/print.h"
#include "utils/math.h"
#include "brain.h"

#include <rerun.hpp>

const int SelfLocator::numberOfSamples = 12;
const float SelfLocator::sigmaAngle = 0.0005;
const float SelfLocator::sigmaDistance = 1;
const float SelfLocator::movedDistWeightRotationNoise = 0.0005;
const float SelfLocator::movedAngleWeightRotationNoise = 0.25;
const float SelfLocator::movedAngleWeightRotationNoiseNotWalking = 0.075;
const float SelfLocator::majorDirTransWeight = 2;
const float SelfLocator::minorDirTransWeight = 1;
const float SelfLocator::validityFactorLandmarkMeasurement = 3;
const int SelfLocator::numberOfConsideredFramesForValidity = 60;
const float SelfLocator::minValidityForSuperbLocalizationQuality = 0.5;
const float SelfLocator::maxTranslationDeviationForSuperbLocalizationQuality = 100.0;
const Angle SelfLocator::maxRotationalDeviationForSuperbLocalizationQuality =
    Angle((30 / 180.f) * pi);
const Pose2f SelfLocator::filterProcessDeviation = Pose2f(0.002, 2.0, 2.0);
const Pose2f SelfLocator::odometryDeviation = Pose2f(0.3, 0.2, 0.2);
const Vector2f SelfLocator::odometryRotationDeviation = Vector2f(0.00157, 0.00157);

void UKFRobotPoseHypothesis::init(const Pose2f& pose, const Pose2f& poseDeviation, int id,
                                  float validity) {
  this->id = id;
  this->validity = validity;
  mean << pose.translation.x(), pose.translation.y(), pose.rotation;
  cov = Matrix3f::Zero();
  cov(0, 0) = sqr(poseDeviation.translation.x());
  cov(1, 1) = sqr(poseDeviation.translation.y());
  cov(2, 2) = sqr(poseDeviation.rotation);
  for (int i = 0; i < 7; ++i) sigmaPoints[i] = Vector3f::Zero();
}

void UKFRobotPoseHypothesis::mirror() {
  const Pose2f newPose = Pose2f(pi) + getPose();
  mean.x() = newPose.translation.x();
  mean.y() = newPose.translation.y();
  mean.z() = newPose.rotation;
}

void UKFRobotPoseHypothesis::updateValidity(int frames, float currentValidity) {
  validity = (validity * (frames - 1) + currentValidity) / frames;
}

void UKFRobotPoseHypothesis::computeWeightingBasedOnValidity(float baseValidityWeighting) {
  weighting = baseValidityWeighting + (1.f - baseValidityWeighting) * validity;
}

void UKFRobotPoseHypothesis::invalidate() {
  validity = 0.f;
}

float UKFRobotPoseHypothesis::getCombinedVariance() const {
  return std::max(cov(0, 0), cov(1, 1)) * cov(2, 2);
}

void UKFRobotPoseHypothesis::updateByLandmark(const RegisteredLandmark& landmark) {
  landmarkSensorUpdate(landmark.model, landmark.percept, landmark.covPercept);
}

SelfLocator::SelfLocator(Brain* brain, const FieldDimensions& fd) : brain(brain) {
  int nextSampleId = 0;

  // Create sample set with samples at the typical walk-in positions
  samples = new SampleSet<UKFRobotPoseHypothesis>(numberOfSamples);
  for (int i = 0; i < samples->size(); ++i) {
    samples->at(i).init(
        {-M_PI / 2, static_cast<float>(((-fd.length / 2) + (-fd.circleRadius)) / 2),
         static_cast<float>(fd.width / 2 + 0.5)},
        {M_PI / 6, static_cast<float>(abs((-fd.length / 2) - (-fd.circleRadius)) / 2), 0.5},
        nextSampleId++, 0.5f);
  }
  goalPosts = {
      Vector2f(fd.length / 2, fd.goalWidth / 2),    // left post of left goal
      Vector2f(fd.length / 2, -fd.goalWidth / 2),   // right post of left goal
      Vector2f(-fd.length / 2, -fd.goalWidth / 2),  // left post of right goal
      Vector2f(-fd.length / 2, fd.goalWidth / 2)    // right post of right goal
  };
  xMarkers = {Vector2f(0.0, -fd.circleRadius), Vector2f(0.0, fd.circleRadius)};
  penaltyMarkers = {Vector2f(fd.length / 2 - fd.penaltyDist, 0.0),
                    Vector2f(-fd.length / 2 + fd.penaltyDist, 0.0)};
  odomInitialized = false;
}

Pose2f SelfLocator::getPose() {
  UKFRobotPoseHypothesis& bestSample = getMostValidSample();
  idOfLastBestSample = bestSample.id;
  return bestSample.getPose();
};

bool SelfLocator::isGood() {
  UKFRobotPoseHypothesis& bestSample = getMostValidSample();
  Matrix3f cov = bestSample.getCov();
  const float transStd = std::sqrt(std::max(cov(0, 0), cov(1, 1)));
  const float rotStd = std::sqrt(cov(2, 2));
  // prtDebug("Validity " + to_string(bestSample.validity) + " translational sd " +
  // to_string(translationalStandardDeviation)
  //          + " rotational sd " + to_string(rotationalStandardDeviation));
  // if(bestSample.validity >= minValidityForSuperbLocalizationQuality &&
  //    translationalStandardDeviation < maxTranslationDeviationForSuperbLocalizationQuality &&
  //    rotationalStandardDeviation < maxRotationalDeviationForSuperbLocalizationQuality) {

  brain->log->log("localization/validity",
                  rerun::TextLog(format("validity: %.2f  trans std: %.2f  rot std: %.2f",
                                        bestSample.validity, transStd, rotStd)));

  bool isGood = (bestSample.validity >= minValidityForSuperbLocalizationQuality);
  return isGood;
};

SelfLocator::~SelfLocator() {
  delete samples;
}

void SelfLocator::motionUpdate(const Pose2D& robotToOdom) {
  Pose2f newOdometryData = Pose2f(robotToOdom.theta, robotToOdom.x, robotToOdom.y);
  Pose2f odometryOffset = newOdometryData - lastOdometryData;
  lastOdometryData = newOdometryData;
  if (!odomInitialized) {
    odomInitialized = true;
    return;
  }

  const float distance = odometryOffset.translation.norm();
  Matrix3f odometryOffsetCovariance;
  odometryOffsetCovariance.setZero();
  odometryOffsetCovariance(0, 0) = sigmaDistance * sigmaDistance * distance;
  odometryOffsetCovariance(1, 1) = sigmaDistance * sigmaDistance * distance;
  const float x = odometryOffset.translation.x();
  const float y = odometryOffset.translation.y();
  odometryOffsetCovariance += (Matrix3f() << y * y / 3.0f, -x * y / 3.0f, -y / 2.0f, -x * y / 3.0f,
                               x * x / 3.0f, x / 2.0f, -y / 2.0f, x / 2.0f, 1.0f)
                                  .finished() *
                              sigmaAngle * sigmaAngle * distance;

  const float transX = odometryOffset.translation.x();
  const float transY = odometryOffset.translation.y();
  const float angle = std::abs(odometryOffset.rotation);
  const float angleWeightNoise = movedAngleWeightRotationNoise;

  // Precalculate rotational error that has to be adapted to all samples
  const float rotError =
      std::max(distance * movedDistWeightRotationNoise, angle * angleWeightNoise);

  // pre-calculate translational error that has to be adapted to all samples
  const float transXError =
      std::max(std::abs(transX * majorDirTransWeight), std::abs(transY * minorDirTransWeight));
  const float transYError =
      std::max(std::abs(transY * majorDirTransWeight), std::abs(transX * minorDirTransWeight));

  // update samples
  for (int i = 0; i < numberOfSamples; ++i) {
    const Vector2f transOffset((transX - transXError) + (2 * transXError) * Random::uniform(),
                               (transY - transYError) + (2 * transYError) * Random::uniform());
    const float rotationOffset = odometryOffset.rotation + Random::uniform(-rotError, rotError);

    samples->at(i).motionUpdate(Pose2f(rotationOffset, transOffset), filterProcessDeviation,
                                odometryDeviation, odometryRotationDeviation);
  }
}

void SelfLocator::sensorUpdate(const std::vector<GameObject>& detectedGoalPosts,
                               const std::vector<GameObject>& detectedMarkings) {
  auto filterByLabel = [](const std::vector<GameObject>& v, const std::string& label) {
    std::vector<GameObject> out;
    std::copy_if(v.begin(), v.end(), std::back_inserter(out),
                 [&](const GameObject& obj) { return obj.label == label; });
    return out;
  };

  const auto detectedXMarkers = filterByLabel(detectedMarkings, "XCross");
  const auto detectedLMarkers = filterByLabel(detectedMarkings, "LCross");
  const auto detectedTMarkers = filterByLabel(detectedMarkings, "TCross");
  const auto detectedPenaltyPoints = filterByLabel(detectedMarkings, "PenaltyPoint");

  // array of paired vectors
  const std::vector<GameObject>* detectedArrays[] = {
      &detectedGoalPosts, &detectedXMarkers,      &detectedLMarkers,
      &detectedTMarkers,  &detectedPenaltyPoints,
  };
  const std::vector<Vector2f>* groundTruthArrays[] = {
      &goalPosts, &xMarkers, &lMarkers, &tMarkers, &penaltyMarkers,
  };
  constexpr size_t nTypes = sizeof(detectedArrays) / sizeof(detectedArrays[0]);

  size_t totalDetectedMarkers = 0;
  for (size_t i = 0; i < nTypes; ++i) {
    totalDetectedMarkers += detectedArrays[i]->size();
  }

  if (totalDetectedMarkers == 0) return;

  // compute samples' validity
  for (int i = 0; i < numberOfSamples; ++i) {
    auto& sample = samples->at(i);
    const Pose2f samplePose = sample.getPose();
    std::vector<RegisteredLandmark> landmarks;
    int numRegisteredLandmarks = 0;

    for (size_t t = 0; t < nTypes; ++t) {
      if (!detectedArrays[t]->empty()) {
        registerLandmarks(samplePose, *detectedArrays[t], *groundTruthArrays[t], landmarks);
        numRegisteredLandmarks = landmarks.size();
      }
    }

    for (const auto& landmark : landmarks) {
      sample.updateByLandmark(landmark);
    }

    const float currentValidity = (static_cast<float>(landmarks.size()) / totalDetectedMarkers);
    sample.updateValidity(numberOfConsideredFramesForValidity, currentValidity);
  }
}

void SelfLocator::registerLandmarks(const Pose2f& samplePose,
                                    const std::vector<GameObject>& detectedObjs,
                                    const std::vector<Vector2f>& groundTruthObjs,
                                    std::vector<RegisteredLandmark>& landmarks) {
  int numDetected = detectedObjs.size();
  int numGroundTruth = groundTruthObjs.size();
  int maxDim = std::max(numDetected, numGroundTruth);

  // Initialize cost matrix with large values (padding with high costs)
  MatrixXd costMatrix = MatrixXd::Constant(maxDim, maxDim, std::numeric_limits<double>::infinity());

  // Fill cost matrix with distances
  for (int i = 0; i < numDetected; ++i) {
    double x, y, z = 0;
    transCoord(detectedObjs[i].posToRobot.x, detectedObjs[i].posToRobot.y, 0,
               samplePose.translation.x(), samplePose.translation.y(), samplePose.rotation, x, y,
               z);
    for (int j = 0; j < numGroundTruth; ++j) {
      costMatrix(i, j) = (Vector2f(x, y) - groundTruthObjs[j]).norm();
    }
  }

  std::vector<int> assignment;
  int numRegistered = 0;
  double totalCost = solveAssignment(costMatrix, assignment, 5.0);
  for (int i = 0; i < detectedObjs.size(); ++i) {
    if (assignment[i] != -1) {
      RegisteredLandmark newLandmark;
      newLandmark.percept = Vector2f(detectedObjs[i].posToRobot.x, detectedObjs[i].posToRobot.y);
      newLandmark.model = groundTruthObjs[assignment[i]];
      newLandmark.covPercept = (Matrix2f() << 1.f, 0.f, 0.f, 1.f).finished();
      landmarks.push_back(newLandmark);
      numRegistered++;
    }
  }
}

double SelfLocator::solveAssignment(const MatrixXd& costMatrix, std::vector<int>& assignment,
                                    const double threshold) {
  // Modified Hungarian Algorithm

  const double INF = std::numeric_limits<double>::infinity();
  int n = (int)costMatrix.rows();

  // Pad the cost matrix with INF to make it square and apply the threshold
  MatrixXd cost = MatrixXd::Constant(n, n, INF);
  cost.block(0, 0, n, n) = costMatrix.unaryExpr([threshold](double val) {
    return (val <= threshold) ? val : std::numeric_limits<double>::infinity();
  });

  // stringstream ss;
  // ss << cost;
  // prtDebug(ss.str());

  assignment.resize(n, -1);

  std::vector<double> u(n + 1), v(n + 1), p(n + 1), way(n + 1);
  for (int i = 1; i <= n; i++) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(n + 1, INF);
    std::vector<bool> used(n + 1, false);
    while (true) {
      used[j0] = true;
      int i0 = p[j0], j1 = 0;
      double delta = INF;
      for (int j = 1; j <= n; j++) {
        if (!used[j]) {
          double cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
      }
      if (delta == INF) {
        // No improvement possible, no augmenting path found.
        // This means not all tasks can be assigned.
        break;
      }
      for (int j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
      if (p[j0] == 0) {
        // Augment the matching
        do {
          int j1 = way[j0];
          p[j0] = p[j1];
          j0 = j1;
        } while (j0);
        break;
      }
    }
  }

  for (int j = 1; j <= n; j++) {
    if (p[j] - 1 >= 0) {
      assignment[p[j] - 1] = j - 1;
    }
  }

  double total_cost = 0;
  for (int i = 0; i < n; i++) {
    if (assignment[i] > -1) {
      total_cost += cost(i, assignment[i]);
      // prtDebug("Assigned " + to_string(i) + " to " + to_string(assignment[i]));
    }
  }
  return total_cost;
}

UKFRobotPoseHypothesis& SelfLocator::getMostValidSample() {
  float validityOfLastBestSample = -1.f;
  UKFRobotPoseHypothesis* lastBestSample = 0;
  if (idOfLastBestSample != -1) {
    for (int i = 0; i < numberOfSamples; ++i) {
      if (samples->at(i).id == idOfLastBestSample) {
        validityOfLastBestSample = samples->at(i).validity;
        lastBestSample = &(samples->at(i));
        break;
      }
    }
  }
  UKFRobotPoseHypothesis* returnSample = &(samples->at(0));
  float maxValidity = -1.f;
  float minVariance = 0.f;  // Initial value does not matter
  for (int i = 0; i < numberOfSamples; ++i) {
    const float val = samples->at(i).validity;
    if (val > maxValidity) {
      maxValidity = val;
      minVariance = samples->at(i).getCombinedVariance();
      returnSample = &(samples->at(i));
    } else if (val == maxValidity) {
      float variance = samples->at(i).getCombinedVariance();
      if (variance < minVariance) {
        maxValidity = val;
        minVariance = variance;
        returnSample = &(samples->at(i));
      }
    }
  }
  if (lastBestSample &&
      returnSample->validity <= validityOfLastBestSample * 1.5f)  // Bonus for stability
    return *lastBestSample;
  else
    return *returnSample;
}