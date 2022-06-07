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
  explicit Estimator(bool verbose);

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
  void SolveOdometry();
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
  double ComputeCurrentLoss();
  ProblemMeta MakeProblem(int landmark_size);
  bool SolveProblem(const ProblemMeta& problem_meta, double lambda, Eigen::VectorXd* delta_x);
  States UpdateStates(const Eigen::VectorXd& delta_x);
  void RollbackStates(const States& states);

  double ComputeInitialLambda(const ProblemMeta& problem);
  bool IsGoodStep(const ProblemMeta& problem, const Eigen::VectorXd& delta_x, double last_lost,
                  double* lambda, double* ni);
  void OptimizeSlideWindow(int max_num_iterations);

  void BackendOptimization();

  // for marginalization
  void MargOldFrame();
  void MargNewFrame();

  bool failureDetection();

  enum SolverFlag { INITIAL, NON_LINEAR };

  enum MarginalizationFlag { MARGIN_OLD = 0, MARGIN_SECOND_NEW = 1 };

  bool verbose_;
  backend::ImuIntrinsic imu_intrinsics_;

  //////////////// OUR SOLVER ///////////////////
  std::shared_ptr<backend::LossFunction> loss_fcn_;
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

  Matrix3d last_R;
  Vector3d last_P;
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

  // MarginalizationInfo *last_marginalization_info;
  vector<double*> last_marginalization_parameter_blocks;

  std::map<int64_t, backend::ImageFrame> all_image_frame;
  backend::IntegrationBase* tmp_pre_integration;

  std::map<uint64_t, Eigen::Vector3d> all_map_points_;
};

}  // namespace vins
