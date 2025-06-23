/**
 * @file UKFRobotPoseHypothesis.h
 *
 * Declaration of a robot pose estimate based on an Unscented Kalman Filter
 *
 * @author Tim Laue
 * @author Colin Graf
 * @author Yuqian Jiang
 */

#pragma once

#include "UKFPose2D.h"
#include "Math/Eigen.h"
#include "SampleSet.h"
#include "types.h"

class Brain;

struct RegisteredLandmark {
  Vector2f percept =
      Vector2f::Zero(); /**< The position of the perceived landmark (relative to the robot) */
  Matrix2f covPercept = Matrix2f::Identity(); /**< The covariance of the landmark measurement */
  Vector2f model =
      Vector2f::Zero(); /**< The position of original landmark (in global coordinates) */
};

/**
 * @class UKFRobotPoseHypothesis
 *
 * Hypothesis of a robot's pose, modeled as an Unscented Kalman Filter.
 * Actual UKF stuff is done by the base class UKFPose2D
 * The pose consists of a position in a 2D plane and an orientation in this plane.
 */
class UKFRobotPoseHypothesis : public UKFPose2D {
 public:
  float weighting; /**< The weighting required for the resampling process. Computation is based on
                      validity and a base weighting. */
  float validity;  /**< The validity represents the average success rate of the measurement matching
                      process. 1 means that all recent measurements are compatible to the sample, 0
                      means that no measurements are compatible.*/
  int id;          /**< Each sample has a unique identifier, which is set at initialization. */

  /** Initializes the members of this sample.
   * @param pose The initial pose
   * @param poseDeviation The initial deviations of the estimates of the different dimensions
   * @param id The unique identifier (caller must make sure that it is really unique)
   * @param validity The initial validity [0,..,1]
   */
  void init(const Pose2f& pose, const Pose2f& poseDeviation, int id, float validity);

  /** The RoboCup field is point-symmetric. Calling this function turns the whole pose by 180
   * degrees around the field's center.*/
  void mirror();

  /** Computes a new validity value based on the current validity and the previous validity.
   * @param frames The old validity is weighted by (frames-1)
   * @param currentValidity The validity of this frame's measurements, weighted by 1
   */
  void updateValidity(int frames, float currentValidity);

  /** Sets the validity to 0, which will automatically lead to 0 weighting, too.
   *  This will cause the sample to be eliminated during the next resampling.
   */
  void invalidate();

  /** Yeah, just like the name says.
   *  Call after measurement / sensor updates.
   *  @param baseValidityWeighting The weighting will have at least this value
   */
  void computeWeightingBasedOnValidity(float baseValidityWeighting);

  /** Returns one variance value by combining x+y+rotational variance in some way*/
  float getCombinedVariance() const;

  /** Update the estimate based on the measurement of a landmark (center circle, penalty mark, ...)
   * @param landmark Yes, the landmark.
   */
  void updateByLandmark(const RegisteredLandmark& landmark);
};

class SelfLocator {
 public:
  SelfLocator(Brain* brain, const FieldDimensions& fd);

  Pose2f getPose();

  bool isGood();

  // TODO: resampling

  /** Integrate odometry offset into hypotheses */
  void motionUpdate(const Pose2D& robotToOdom);

  /** Perform UKF measurement step for all samples */
  void sensorUpdate(const std::vector<GameObject>& detectedGoalPosts,
                    const std::vector<GameObject>& detectedMarkings);

  void logLandmarks();
  void logSamples();

  /** Destructor */
  ~SelfLocator();

 protected:
  double solveAssignment(const MatrixXd& costMatrix, std::vector<int>& assignment,
                         const double threshold);

  void registerLandmarks(const Pose2f& samplePose, const std::vector<GameObject>& detectedObjs,
                         const std::vector<Vector2f>& groundTruthObjs,
                         std::vector<RegisteredLandmark>& landmarks);

 private:
  SampleSet<UKFRobotPoseHypothesis>* samples; /**< Container for all samples. */
  int idOfLastBestSample; /**< Identifier of the best sample of the last frame */

  // landmarks
  std::vector<Vector2f> goalPosts;
  std::vector<Vector2f> xMarkers;
  std::vector<Vector2f> lMarkers;
  std::vector<Vector2f> tMarkers;
  std::vector<Vector2f> penaltyMarkers;

  Pose2f lastOdometryData;
  bool odomInitialized;

  static const int numberOfSamples;
  static const float sigmaAngle;
  static const float sigmaDistance;
  static const float movedDistWeightRotationNoise;
  static const float movedAngleWeightRotationNoise;
  static const float movedAngleWeightRotationNoiseNotWalking;
  static const float majorDirTransWeight;
  static const float minorDirTransWeight;
  static const float validityFactorLandmarkMeasurement;
  static const int numberOfConsideredFramesForValidity;
  static const float minValidityForSuperbLocalizationQuality;
  static const float maxTranslationDeviationForSuperbLocalizationQuality;
  static const Angle maxRotationalDeviationForSuperbLocalizationQuality;
  static const Pose2f filterProcessDeviation;
  static const Pose2f odometryDeviation;
  static const Vector2f odometryRotationDeviation;

  /** Returns a reference to the sample that has the highest validity
   * @return A reference sample
   */
  UKFRobotPoseHypothesis& getMostValidSample();

  Brain* brain;
};
