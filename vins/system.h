#pragma once

#include <stdio.h>
#include <map>
#include <mutex>
#include <queue>
#include <thread>

#include <condition_variable>
#include <fstream>

#include "vins/estimator.h"
#include "vins/proto/vins_config.pb.h"

#include <pangolin/pangolin.h>

// imu for vio
struct IMU_MSG {
  double header;
  Eigen::Vector3d linear_acceleration;
  Eigen::Vector3d angular_velocity;
};
typedef std::shared_ptr<IMU_MSG const> ImuConstPtr;

// image for vio
struct IMG_MSG {
  double header;
  vector<Vector3d> points;
  vector<int> id_of_point;
  vector<float> u_of_point;
  vector<float> v_of_point;
  vector<float> velocity_x_of_point;
  vector<float> velocity_y_of_point;
};
typedef std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> FEATURE_MSG;
typedef std::shared_ptr<FEATURE_MSG const> FeatureConstPtr;
typedef std::shared_ptr<IMG_MSG const> ImgConstPtr;

namespace vins {

class System {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  System(const vins::proto::VinsConfig& vins_config);
  ~System();

  // push in real image and IMU data
  void PubImageData(double dStampSec, cv::Mat& img);
  void PubImuData(double dStampSec, const Eigen::Vector3d& vGyr, const Eigen::Vector3d& vAcc);

  // thread: visual-inertial odometry
  void ProcessBackEnd();

  // save trajectory
  void SavePoseAsTUM(std::string filename);

 private:
  bool verbose_debug = true;
  bool verbose_pub = false;

  // feature tracker
  std::vector<uchar> r_status;
  std::vector<float> r_err;

  feature::FeatureTracker trackerData[feature::NUM_OF_CAM];
  double first_image_time;
  int pub_count = 1;
  bool first_image_flag = true;
  double last_image_time = 0;
  bool init_pub = 0;

  // estimator
  Estimator estimator;

  std::condition_variable con;
  double current_time = -1;
  std::queue<ImuConstPtr> imu_buf;
  std::queue<ImgConstPtr> feature_buf;
  int sum_of_wait = 0;

  std::mutex m_buf;
  std::mutex m_state;
  std::mutex i_buf;
  std::mutex m_estimator;

  double latest_time;
  Eigen::Vector3d tmp_P;
  Eigen::Quaterniond tmp_Q;
  Eigen::Vector3d tmp_V;
  Eigen::Vector3d tmp_Ba;
  Eigen::Vector3d tmp_Bg;
  Eigen::Vector3d acc_0;
  Eigen::Vector3d gyr_0;

  // flags
  bool init_feature = false;
  bool init_imu = 1;
  double last_imu_t = -1;

  std::vector<Eigen::Vector3d> vPath_to_draw;  // history path of body(IMU)

  std::vector<double> corresponding_timestamps;
  std::vector<Eigen::Matrix<double, 3, 4>> keyframe_history;  // record all the past keyframes

  bool bStart_backend;

 private:
  std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>> getMeasurements();
  Vector2d FindVelocity(IMG_MSG* last_features, int idx, double u, double v, double dStampSec);

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
