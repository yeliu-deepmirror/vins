#pragma once

#include "vins/feature/feature_manager.h"
#include "vins/feature/feature_tracker.h"

#include "vins/initialization/initial_alignment.h"
#include "vins/initialization/initial_sfm.h"
#include "vins/initialization/solve_5pts.h"

#include "vins/backend/common/utility.h"
#include "vins/backend/problem.h"
#include "vins/backend/vertex/vertex_inverse_depth.h"
#include "vins/backend/vertex/vertex_pose.h"
#include "vins/backend/vertex/vertex_speedbias.h"

#include "vins/backend/edge/edge_imu.h"
#include "vins/backend/edge/edge_reprojection.h"

#include "vins/parameters.h"

#include <opencv2/core/eigen.hpp>
#include <queue>

namespace vins {

class Estimator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  Estimator();

  void SetParameter(const Sophus::SO3d& ric, const Eigen::Vector3d& tic);

  // interface
  void processIMU(double t, const Vector3d& linear_acceleration, const Vector3d& angular_velocity);

  // original VINS interface for image (which support stereo camera)
  void ProcessImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& image,
                    double header);

  void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d>& _match_points,
                    Vector3d _relo_t, Matrix3d _relo_r);

  Eigen::Matrix<double, 3, 4> GetCurrentCameraPose();

  // internal
  void ClearState(bool bInit = false);
  bool InitialStructure();
  bool visualInitialAlign();
  bool relativePose(Matrix3d& relative_R, Vector3d& relative_T, int& l);
  void slideWindow();
  void SolveOdometry();
  void slideWindowNew();
  void slideWindowOld();
  void optimization();
  void BackendOptimization();

  void ProblemSolve();
  void MargOldFrame();
  void MargNewFrame();

  bool failureDetection();

  enum SolverFlag { INITIAL, NON_LINEAR };

  enum MarginalizationFlag { MARGIN_OLD = 0, MARGIN_SECOND_NEW = 1 };

  //////////////// OUR SOLVER ///////////////////
  MatXX Hprior_;
  VecX bprior_;
  VecX errprior_;
  MatXX Jprior_inv_;

  Eigen::Matrix2d project_sqrt_info_;

  //////////////// OUR SOLVER //////////////////
  SolverFlag solver_flag;
  MarginalizationFlag marginalization_flag;
  Vector3d g;
  MatrixXd Ap[2], backup_A;
  VectorXd bp[2], backup_b;

  Vec7 vPic[feature::NUM_OF_CAM];
  Sophus::SE3d rigid_ic_;

  Vector3d Ps[(feature::WINDOW_SIZE + 1)];
  Vector3d Vs[(feature::WINDOW_SIZE + 1)];
  Matrix3d Rs[(feature::WINDOW_SIZE + 1)];
  Vector3d Bas[(feature::WINDOW_SIZE + 1)];
  Vector3d Bgs[(feature::WINDOW_SIZE + 1)];
  VectorXd vInverseDepth;
  double td;

  Matrix3d back_R0, last_R, last_R0;
  Vector3d back_P0, last_P, last_P0;
  double Headers[(feature::WINDOW_SIZE + 1)];

  backend::IntegrationBase* pre_integrations[(feature::WINDOW_SIZE + 1)];
  Vector3d acc_0, gyr_0;

  vector<double> dt_buf[(feature::WINDOW_SIZE + 1)];
  vector<Vector3d> linear_acceleration_buf[(feature::WINDOW_SIZE + 1)];
  vector<Vector3d> angular_velocity_buf[(feature::WINDOW_SIZE + 1)];

  int frame_count;
  int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

  feature::FeatureManager f_manager;
  MotionEstimator m_estimator;

  bool first_imu;
  bool is_valid, is_key;
  bool failure_occur;

  // point cloud saved
  bool save_cloud_for_show_ = true;
  vector<Vector3d> point_cloud;
  vector<Vector3d> margin_cloud;
  vector<vector<Vector3d>> margin_cloud_cloud;
  vector<Vector3d> key_poses;
  double initial_timestamp;

  void SaveMarginalizedFrameHostedPoints(vins::backend::Problem& problem);
  void UpdateCurrentPointcloud();

  double para_Pose[feature::WINDOW_SIZE + 1][SIZE_POSE];  // x, y, z, qx, qy, qz, qw
  double para_SpeedBias[feature::WINDOW_SIZE + 1]
                       [SIZE_SPEEDBIAS];                // vx, vy, vz, bax, bay, baz, bgx, bgy, bgz
  double para_Ex_Pose[feature::NUM_OF_CAM][SIZE_POSE];  // tic : x, y, z, qx, qy, qz, qw
  double para_Retrive_Pose[SIZE_POSE];                  // not used
  double para_Td[1][1];                                 // td
  double para_Tr[1][1];                                 // tr

  int loop_window_index;

  // MarginalizationInfo *last_marginalization_info;
  vector<double*> last_marginalization_parameter_blocks;

  map<double, backend::ImageFrame> all_image_frame;
  backend::IntegrationBase* tmp_pre_integration;

  // relocalization variable
  bool relocalization_info;
  double relo_frame_stamp;
  double relo_frame_index;
  int relo_frame_local_index;
  vector<Vector3d> match_points;
  double relo_Pose[SIZE_POSE];
  Matrix3d drift_correct_r;
  Vector3d drift_correct_t;
  Vector3d prev_relo_t;
  Matrix3d prev_relo_r;
  Vector3d relo_relative_t;
  Quaterniond relo_relative_q;
  double relo_relative_yaw;
};

}  // namespace vins
