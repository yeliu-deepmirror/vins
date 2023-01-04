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
#include "vins/backend/edge/edge_prior.h"
#include "vins/backend/edge/edge_reprojection.h"

#include <opencv2/core/eigen.hpp>
#include <queue>

namespace vins {

class Estimator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  explicit Estimator(bool verbose);

  void SetParameter(const Sophus::SO3d& ric, const Eigen::Vector3d& tic);

  // interface
  void processIMU(double t, const Vector3d& linear_acceleration, const Vector3d& angular_velocity);

  // original VINS interface for image (which support stereo camera)
  void ProcessImage(const std::map<int, std::vector<std::pair<int, Eigen::Vector3d>>>& image,
                    int64_t header, const std::optional<backend::ImuState>& imu_state);

  // internal
  void ClearState(bool bInit = false);
  bool InitialStructure();
  bool visualInitialAlign();
  bool relativePose(Matrix3d& relative_R, Vector3d& relative_T, int& l);
  void slideWindow();
  void SolveOdometry();
  void slideWindowNew();
  void slideWindowOld(const Eigen::Matrix3d& back_R0, const Eigen::Vector3d& back_P0);
  void optimization();
  void BackendOptimization();

  void ProblemSolve();
  void MargOldFrame();
  void MargNewFrame();

  bool failureDetection();

  enum SolverFlag { INITIAL, NON_LINEAR };

  enum MarginalizationFlag { MARGIN_OLD = 0, MARGIN_SECOND_NEW = 1 };

  bool verbose_;
  bool save_pts_ = true;
  double robust_loss_factor_ = 1.0;
  int gn_iterations_ = 8;
  bool estimate_extrinsics_ = false;
  bool reset_origin_each_iter_ = true;

  //////////////// OUR SOLVER ///////////////////
  void ExtendedPrior(int dim);
  MatXX Hprior_;
  VecX bprior_;
  VecX errprior_;
  MatXX Jprior_inv_;

  //////////////// OUR SOLVER //////////////////
  SolverFlag solver_flag;
  MarginalizationFlag marginalization_flag;
  Vector3d gravity_;
  backend::ImuIntrinsic imu_intrinsics_;
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

  Matrix3d last_R, last_R0;
  Vector3d last_P, last_P0;
  int64_t Headers[(feature::WINDOW_SIZE + 1)];

  backend::IntegrationBase* pre_integrations[(feature::WINDOW_SIZE + 1)];
  Vector3d acc_0, gyr_0;

  vector<double> dt_buf[(feature::WINDOW_SIZE + 1)];
  vector<Vector3d> linear_acceleration_buf[(feature::WINDOW_SIZE + 1)];
  vector<Vector3d> angular_velocity_buf[(feature::WINDOW_SIZE + 1)];

  int frame_count;

  feature::FeatureManager f_manager;
  MotionEstimator m_estimator;

  bool first_imu;
  bool failure_occur;

  // point cloud saved
  int64_t initial_timestamp;

  std::map<int, Eigen::Vector3d> all_map_points_;
  std::map<int64_t, backend::ImageFrame> all_image_frame;
  backend::IntegrationBase* tmp_pre_integration;

  Eigen::Matrix2d project_sqrt_info_;
  Eigen::Matrix<double, 1, 1> depth_information_;
};

}  // namespace vins
