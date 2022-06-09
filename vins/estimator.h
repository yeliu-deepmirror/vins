#pragma once

#include "vins/feature/feature_manager.h"
#include "vins/feature/feature_tracker.h"

#include "vins/initialization/initial_alignment.h"
#include "vins/initialization/initial_sfm.h"
#include "vins/initialization/solve_5pts.h"

#include "vins/backend/common/cost_function.h"
#include "vins/backend/common/loss_function.h"
#include "vins/backend/common/tic_toc.h"
#include "vins/parameters.h"

#include <opencv2/core/eigen.hpp>
#include <queue>

namespace vins {

class Estimator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  Estimator(bool verbose, double focus);

  void SetParameter(const Sophus::SO3d& ric, const Eigen::Vector3d& tic);

  // interface
  void processIMU(double t, const Vector3d& linear_acceleration, const Vector3d& angular_velocity);

  // original VINS interface for image (which support stereo camera)
  void ProcessImage(const std::map<uint64_t, std::vector<std::pair<int, Eigen::Vector3d>>>& image,
                    int64_t header);

  // internal
  void ClearState(bool bInit = false);
  bool InitialStructure();
  bool VisualInitialAlign();
  bool relativePose(Matrix3d& relative_R, Vector3d& relative_T, int& l);
  void slideWindow();
  int SolveOdometry();
  void slideWindowNew();
  void slideWindowOld(const Eigen::Matrix3d& back_R0, const Eigen::Vector3d& back_P0);

  // for optimization process
  struct ProblemMeta {
    double total_lost;
    Eigen::MatrixXd h_matrix;
    Eigen::VectorXd b_vec;
  };
  struct States {
    Sophus::SE3d rigid_ic;
    Eigen::Vector3d Ps[(feature::WINDOW_SIZE + 1)];
    Eigen::Vector3d Vs[(feature::WINDOW_SIZE + 1)];
    Eigen::Matrix3d Rs[(feature::WINDOW_SIZE + 1)];
    Eigen::Vector3d Bas[(feature::WINDOW_SIZE + 1)];
    Eigen::Vector3d Bgs[(feature::WINDOW_SIZE + 1)];

    Eigen::VectorXd bprior;
    Eigen::VectorXd errprior;
    std::vector<double> depths;
  };
  std::pair<int, int> RejectOutliers(double threshold);
  double ComputeCurrentLoss(std::vector<double>* sqr_residuals);
  ProblemMeta MakeProblem(int landmark_size);
  States UpdateStates(const Eigen::VectorXd& delta_x);
  void RollbackStates(const States& states);
  double OptimizeSlideWindow(int max_num_iterations, bool veb = false);

  int BackendOptimization();

  // for marginalization
  void MargOldFrame();
  void MargNewFrame();

  bool FailureDetection(int tracking_cnt);

  enum SolverFlag { INITIAL, NON_LINEAR };

  enum MarginalizationFlag { MARGIN_OLD = 0, MARGIN_SECOND_NEW = 1 };

  bool verbose_;
  backend::ImuIntrinsic imu_intrinsics_;

  //////////////// OUR SOLVER ///////////////////
  std::shared_ptr<backend::LossFunction> loss_fcn_;

  bool valid_prior_ = false;
  MatXX Hprior_;
  VecX bprior_;
  VecX errprior_;
  MatXX Jprior_inv_;
  void ExtendedPrior(int dim);

  Eigen::Matrix2d project_sqrt_info_;

  //////////////// OUR SOLVER //////////////////
  SolverFlag solver_flag;
  MarginalizationFlag marginalization_flag;
  Vector3d g;

  Sophus::SE3d rigid_ic_;
  int64_t Headers[(feature::WINDOW_SIZE + 1)];
  Vector3d Ps[(feature::WINDOW_SIZE + 1)];
  Vector3d Vs[(feature::WINDOW_SIZE + 1)];
  Matrix3d Rs[(feature::WINDOW_SIZE + 1)];
  Vector3d Bas[(feature::WINDOW_SIZE + 1)];
  Vector3d Bgs[(feature::WINDOW_SIZE + 1)];

  backend::IntegrationBase* pre_integrations[(feature::WINDOW_SIZE + 1)];
  Vector3d acc_0, gyr_0;

  std::vector<double> dt_buf[(feature::WINDOW_SIZE + 1)];
  std::vector<Eigen::Vector3d> linear_acceleration_buf[(feature::WINDOW_SIZE + 1)];
  std::vector<Eigen::Vector3d> angular_velocity_buf[(feature::WINDOW_SIZE + 1)];

  int frame_count;

  feature::FeatureManager f_manager;
  MotionEstimator m_estimator;

  bool first_imu;
  bool failure_occur;

  std::map<int64_t, backend::ImageFrame> all_image_frame;
  backend::IntegrationBase* tmp_pre_integration;
  std::map<uint64_t, Eigen::Vector3d> all_map_points_;

  double depth_weight_ = 10.0;
};

void UpdateProblemMatrix(const std::vector<Eigen::MatrixXd>& jacobians,
                         const std::vector<int>& indices, const std::vector<int>& dimensions,
                         const Eigen::VectorXd& residual, const Eigen::MatrixXd& information,
                         const Eigen::MatrixXd& robust_information, double drho,
                         Eigen::MatrixXd* h_mat, Eigen::VectorXd* b_vec);
}  // namespace vins
