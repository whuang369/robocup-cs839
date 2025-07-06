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

const Pose2f SelfLocator::defaultPoseDeviation = Pose2f(0.3, 0.5, 0.5);
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
  for (int i = 0; i < 7; ++i) {
    sigmaPoints[i] = Vector3f::Zero();
  }
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
  nextSampleId = 0;

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
  xMarkers = {Vector2f(0.0, fd.circleRadius), Vector2f(0.0, -fd.circleRadius)};
  tMarkers = {Vector2f(0.0, fd.width / 2),
              Vector2f(0.0, -fd.width / 2),
              Vector2f(fd.length / 2, fd.penaltyAreaWidth / 2),
              Vector2f(fd.length / 2, -fd.penaltyAreaWidth / 2),
              Vector2f(-fd.length / 2, fd.penaltyAreaWidth / 2),
              Vector2f(-fd.length / 2, -fd.penaltyAreaWidth / 2),
              Vector2f(fd.length / 2, -fd.goalAreaWidth / 2),
              Vector2f(fd.length / 2, fd.goalAreaWidth / 2),
              Vector2f(-fd.length / 2, -fd.goalAreaWidth / 2),
              Vector2f(-fd.length / 2, fd.goalAreaWidth / 2)};

  float centerToPenalty = fd.length / 2 - fd.penaltyAreaLength;
  float centerToGoalArea = fd.length / 2 - fd.goalAreaLength;
  lMarkers = {Vector2f(fd.length / 2, fd.width / 2),
              Vector2f(fd.length / 2, -fd.width / 2),
              Vector2f(-fd.length / 2, fd.width / 2),
              Vector2f(-fd.length / 2, -fd.width / 2),
              Vector2f(centerToPenalty, fd.penaltyAreaWidth / 2),
              Vector2f(centerToPenalty, -fd.penaltyAreaWidth / 2),
              Vector2f(-centerToPenalty, fd.penaltyAreaWidth / 2),
              Vector2f(-centerToPenalty, -fd.penaltyAreaWidth / 2),
              Vector2f(centerToGoalArea, fd.goalAreaWidth / 2),
              Vector2f(centerToGoalArea, -fd.goalAreaWidth / 2),
              Vector2f(-centerToGoalArea, fd.goalAreaWidth / 2),
              Vector2f(-centerToGoalArea, -fd.goalAreaWidth / 2)};
  penaltyMarkers = {Vector2f(fd.length / 2 - fd.penaltyDist, 0.0),
                    Vector2f(-fd.length / 2 + fd.penaltyDist, 0.0)};
  odomInitialized = false;

  logLandmarks();
}

void SelfLocator::init(const FieldDimensions& fd, std::string& attackSide, float startPos) {
  startPos = std::clamp(startPos, -1.0f, 1.0f);
  const float sideSign = (attackSide == "left") ? -1.0f : 1.0f;
  const float initX = sideSign * static_cast<float>(fd.length / 2) * startPos;
  const float initY = -sideSign * (static_cast<float>(fd.width / 2) + 0.5);
  const float initTheta = sideSign * M_PI_2;

  float thetaNoise = M_PI / 6;
  float xNoise = static_cast<float>(fd.length) * 0.1f;
  float yNoise = 0.5;

  if (!samples) {
    samples = new SampleSet<UKFRobotPoseHypothesis>(numberOfSamples);
  }

  for (int i = 0; i < numberOfSamples; ++i) {
    samples->at(i).init({initTheta, initX, initY}, {thetaNoise, xNoise, yNoise}, nextSampleId++,
                        0.5f);
  }
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

  bool isGood = (bestSample.validity >= minValidityForSuperbLocalizationQuality);

  brain->log->log(
      "localization/validity",
      rerun::TextLog(format("isGood: %d  validity: %.2f  trans std: %.2f  rot std: %.2f", isGood,
                            bestSample.validity, transStd, rotStd)));
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

  const float transX = odometryOffset.translation.x();
  const float transY = odometryOffset.translation.y();
  const float dist = odometryOffset.translation.norm();
  const float angle = std::abs(odometryOffset.rotation);
  const float angleWeightNoise = movedAngleWeightRotationNoise;

  // Precalculate rotational error that has to be adapted to all samples
  const float rotError = std::max(dist * movedDistWeightRotationNoise, angle * angleWeightNoise);

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

  // sensor update for each sample
  for (int i = 0; i < numberOfSamples; ++i) {
    auto& sample = samples->at(i);
    const Pose2f samplePose = sample.getPose();
    std::vector<RegisteredLandmark> landmarks;
    int numRegisteredLandmarks = 0;

    // register landmarks
    for (size_t t = 0; t < nTypes; ++t) {
      if (!detectedArrays[t]->empty()) {
        registerLandmarks(samplePose, *detectedArrays[t], *groundTruthArrays[t], landmarks);
        numRegisteredLandmarks = landmarks.size();
      }
    }

    // pose update based on the perceived landmarks
    for (const auto& landmark : landmarks) {
      sample.updateByLandmark(landmark);
    }

    // update sample validity
    const float currentValidity = (static_cast<float>(landmarks.size()) / totalDetectedMarkers);
    sample.updateValidity(numberOfConsideredFramesForValidity, currentValidity);
  }
}

void SelfLocator::registerLandmarks(const Pose2f& samplePose,
                                    const std::vector<GameObject>& detectedObjs,
                                    const std::vector<Vector2f>& groundTruthObjs,
                                    std::vector<RegisteredLandmark>& landmarks) {
  const float MAX_ASSOCIATION_DISTANCE = 1.0f;

  if (groundTruthObjs.empty() || detectedObjs.empty()) return;

  auto computeCovariance = [&](const GameObject& obj) {
    float sigmaX = 0.01f * sqr(static_cast<float>(obj.range));
    float confidence = std::max(static_cast<float>(obj.confidence), 0.1f);
    float sigmaY = 0.01f / confidence * sqr(static_cast<float>(obj.range));

    Matrix2f cov = Matrix2f::Zero();
    cov(0, 0) = sqr(sigmaX);
    cov(1, 1) = sqr(sigmaY);
    return cov;
  };

  for (const auto& obj : detectedObjs) {
    Vector2f perceptOnField = samplePose * Vector2f(obj.posToRobot.x, obj.posToRobot.y);

    float minDistSq = -1.f;
    int bestMatchIndex = -1;
    for (int i = 0; i < groundTruthObjs.size(); ++i) {
      float distSq = (perceptOnField - groundTruthObjs[i]).squaredNorm();
      if (bestMatchIndex == -1 || distSq < minDistSq) {
        minDistSq = distSq;
        bestMatchIndex = i;
      }
    }

    // validation gate
    if (bestMatchIndex != -1 && minDistSq < sqr(MAX_ASSOCIATION_DISTANCE)) {
      RegisteredLandmark newLandmark;
      newLandmark.percept = Vector2f(obj.posToRobot.x, obj.posToRobot.y);
      newLandmark.model = groundTruthObjs[bestMatchIndex];
      // newLandmark.covPercept = computeCovariance(obj);
      newLandmark.covPercept = (Matrix2f() << 1.0f, 0.0f, 0.0f, 1.0f).finished();
      landmarks.push_back(newLandmark);
    }
  }
}

void SelfLocator::resampling() {
  float totalValidity = 0.f;
  for (int i = 0; i < numberOfSamples; ++i) {
    totalValidity += samples->at(i).validity;
  }
  float averageWeighting = totalValidity / numberOfSamples;
  if (averageWeighting == 0.f) return;

  // resample
  UKFRobotPoseHypothesis* oldSet = samples->swap();
  const float weightingBetweenTwoDrawnSamples = totalValidity / numberOfSamples;
  float nextPos = Random::uniform() * weightingBetweenTwoDrawnSamples;
  float currentSum = 0.f;

  int replacements = 0;
  int j = 0;
  for (int i = 0; i < numberOfSamples; ++i) {
    currentSum += oldSet[i].weighting;
    int replicationCount = 0;
    while (currentSum > nextPos && j < numberOfSamples) {
      samples->at(j) = oldSet[i];
      if (replicationCount) {
        samples->at(j).id = nextSampleId++;
        replacements++;
      }
      replicationCount++;
      j++;
      nextPos += weightingBetweenTwoDrawnSamples;
    }
  }

  // fill up missing samples
  for (; j < numberOfSamples; ++j) {
    const Pose2f pose = samples->at(0).getPose();
    samples->at(j).init(pose, defaultPoseDeviation, nextSampleId++, averageWeighting);
  }
}

UKFRobotPoseHypothesis& SelfLocator::getMostValidSample() {
  float validityOfLastBestSample = -1.f;
  UKFRobotPoseHypothesis* lastBestSample = nullptr;
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
  float maxValidity = -1.0f;
  float minVariance = 0.0f;  // Initial value does not matter
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
  if (lastBestSample && returnSample->validity <= validityOfLastBestSample * 1.5f) {
    return *lastBestSample;
  } else {
    return *returnSample;
  }
}

void SelfLocator::logLandmarks() {
  // rerun logging
  std::vector<rerun::Vec2D> xPoints;
  for (const auto& marker : xMarkers) {
    xPoints.emplace_back(rerun::Vec2D{marker.x(), marker.y()});
  }
  std::vector<rerun::Vec2D> lPoints;
  for (const auto& marker : lMarkers) {
    lPoints.emplace_back(rerun::Vec2D{marker.x(), marker.y()});
  }
  std::vector<rerun::Vec2D> tPoints;
  for (const auto& marker : tMarkers) {
    tPoints.emplace_back(rerun::Vec2D{marker.x(), marker.y()});
  }
  std::vector<rerun::Vec2D> penaltyPoints;
  for (const auto& marker : penaltyMarkers) {
    penaltyPoints.emplace_back(rerun::Vec2D{marker.x(), marker.y()});
  }

  brain->log->log("field/landmarks/x_markers",
                  rerun::Points2D(xPoints).with_colors(0x00FFFFFF).with_radii(0.1));
  brain->log->log("field/landmarks/l_markers",
                  rerun::Points2D(lPoints).with_colors(0xFFFF00FF).with_radii(0.1));
  brain->log->log("field/landmarks/t_markers",
                  rerun::Points2D(tPoints).with_colors(0x00FF00FF).with_radii(0.1));
  brain->log->log("field/landmarks/penalty_markers",
                  rerun::Points2D(penaltyPoints).with_colors(0xFFD700FF).with_radii(0.1));
}

void SelfLocator::logSamples() {
  for (int i = 0; i < numberOfSamples; ++i) {
    const Pose2f pose = samples->at(i).getPose();
    float x = pose.translation.x();
    float y = pose.translation.y();
    float theta = pose.rotation;
    string id = std::to_string(samples->at(i).id);
    brain->log->log("field/pose_samples/" + id,
                    rerun::Points2D({{x, -y}, {x + 0.05 * cos(theta), -y - 0.05 * sin(theta)}})
                        .with_radii({0.1, 0.05})
                        .with_colors({0xFFAA00FF, 0xFFFF00FF}));
  }
}