#pragma once

#include <map>
#include <mutex>
#include <queue>

#include "vins/estimator.h"
#include "vins/proto/vins_config.pb.h"
#include "sophus/se3.hpp"

#include <pangolin/pangolin.h>

namespace vins {

class System {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  explicit System(const vins::proto::VinsConfig& vins_config);
  ~System();

  // push in real image and IMU data
  // depth should be float scalar
  bool PublishImageData(int64_t timestamp, cv::Mat& img, cv::Mat& depth);
  bool PublishImuData(int64_t timestamp, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr);
  void ShowTrack(cv::Mat* image);

  // private:
  const vins::proto::VinsConfig vins_config_;

  feature::FeatureTracker feature_tracker_;
  Estimator estimator_;

  double current_time_ = -1;

  Eigen::Vector3d latest_acc_;
  Eigen::Vector3d latest_gyr_;

  // buffer the trajectory
  std::vector<Eigen::Vector3d> frame_positions_;  // history path of body(IMU
  std::vector<Eigen::Matrix<double, 3, 4>> keyframe_history;  // record all the past keyframes

 public:
  // functions for visualization
  void Draw();

  float mCameraSize = 0.5;
  float mCameraLineWidth = 1;

  void DrawCamera();

  void DrawKeyframe(Eigen::Matrix3d matrix_t, Eigen::Vector3d pVector_t);
  void DrawKeyframe(Eigen::Matrix<double, 3, 4> Twc);

  void GetOpenGLCameraMatrix(Eigen::Matrix3d matrix, Eigen::Vector3d pVector,
                             pangolin::OpenGlMatrix& M, bool if_inv);

  void GetOpenGLMatrixCamera(pangolin::OpenGlMatrix& M, Eigen::Matrix<double, 3, 4> Twc);
};

}  // namespace vins
