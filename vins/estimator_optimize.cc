
#include "vins/estimator.h"

namespace vins {

std::pair<int, int> Estimator::RejectOutliers(double threshold) {
  int outlier_cnt = 0;
  int inlier_cnt = 0;
  for (auto& it_per_id : f_manager.feature) {
    if (!it_per_id.Valid()) continue;

    int imu_i = it_per_id.start_frame;
    const Eigen::Vector3d& pt_1 = it_per_id.feature_per_frame[0].point;
    double inv_depth_i = 1.0 / it_per_id.estimated_depth;

    size_t inlier_end = it_per_id.feature_per_frame.size();
    for (size_t i = 1; i < it_per_id.feature_per_frame.size(); i++) {
      const Eigen::Vector3d& pt_2 = it_per_id.feature_per_frame[i].point;
      int imu_j = imu_i + i;
      Eigen::VectorXd residual = backend::ComputeVisualResidual(
          rigid_ic_.translation(), rigid_ic_.so3().matrix(), Ps[imu_i], Rs[imu_i], Ps[imu_j],
          Rs[imu_j], inv_depth_i, pt_1, pt_2);

      if (residual.squaredNorm() > threshold) {
        inlier_end = i;
        break;
      }
    }

    if (inlier_end < 3) {
      it_per_id.solve_flag = 2;
      outlier_cnt++;
    } else {
      inlier_cnt++;
    }

    it_per_id.feature_per_frame.resize(inlier_end);
  }
  return std::make_pair(outlier_cnt, inlier_cnt);
}

// TODO(yeliu) : refine using :
// github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/trust_region_minimizer.cc#L415
double Estimator::ComputeCurrentLoss(std::vector<double>* sqr_residuals) {
  double total_lost = 0;
  sqr_residuals->clear();
  // for imu factors
  for (int i = 0; i < feature::WINDOW_SIZE; i++) {
    int j = i + 1;
    if (pre_integrations[j]->sum_dt > 10.0) continue;

    Eigen::VectorXd residual;
    backend::ComputeImuJacobian(Ps[i], Eigen::Quaterniond(Rs[i]), Vs[i], Bas[i], Bgs[i], Ps[j],
                                Eigen::Quaterniond(Rs[j]), Vs[j], Bas[j], Bgs[j],
                                pre_integrations[j], nullptr, nullptr, nullptr, nullptr, &residual);
    Eigen::MatrixXd information = pre_integrations[j]->covariance.inverse();
    total_lost += residual.transpose() * information * residual;
  }

  // for visual factors
  Eigen::MatrixXd vis_information = project_sqrt_info_.transpose() * project_sqrt_info_;

  // for all the features
  for (auto& it_per_id : f_manager.feature) {
    if (!it_per_id.Valid()) continue;

    int imu_i = it_per_id.start_frame;
    const Eigen::Vector3d& pt_1 = it_per_id.feature_per_frame[0].point;
    double inv_depth_i = 1.0 / it_per_id.estimated_depth;

    if (inv_depth_i < 0) continue;

    if (it_per_id.depth_gt.has_value()) {
      // add inverse depth ground truth value factor
      double inv_depth_gt = 1.0 / it_per_id.depth_gt.value();
      double residual = inv_depth_i - inv_depth_gt;
      total_lost += residual * residual * depth_weight_;
    }

    for (size_t i = 1; i < it_per_id.feature_per_frame.size(); i++) {
      const Eigen::Vector3d& pt_2 = it_per_id.feature_per_frame[i].point;
      int imu_j = imu_i + i;
      Eigen::VectorXd residual = backend::ComputeVisualResidual(
          rigid_ic_.translation(), rigid_ic_.so3().matrix(), Ps[imu_i], Rs[imu_i], Ps[imu_j],
          Rs[imu_j], inv_depth_i, pt_1, pt_2);

      sqr_residuals->emplace_back(residual.squaredNorm());
      // update robust loss fcn
      double cost = residual.transpose() * vis_information * residual;
      total_lost += loss_fcn_->Compute(cost)[0];
    }
  }
  // add prior
  if (errprior_.rows() > 0) total_lost += errprior_.norm();
  return total_lost * 0.5;
}

void Estimator::RollbackStates(const Estimator::States& backup) {
  if (ESTIMATE_EXTRINSIC) {
    rigid_ic_ = backup.rigid_ic;
  }
  for (int i = 0; i < feature::WINDOW_SIZE + 1; i++) {
    Ps[i] = backup.Ps[i];
    Rs[i] = backup.Rs[i];
    Vs[i] = backup.Vs[i];
    Bas[i] = backup.Bas[i];
    Bgs[i] = backup.Bgs[i];
  }

  bprior_ = backup.bprior;
  errprior_ = backup.errprior;

  int idx = 0;
  for (auto& it_per_id : f_manager.feature) {
    if (!it_per_id.Valid()) continue;
    it_per_id.estimated_depth = backup.depths[idx++];
  }
}

Estimator::States Estimator::UpdateStates(const Eigen::VectorXd& delta_x) {
  static int pose_dim = (feature::WINDOW_SIZE + 1) * 15 + 6;
  States backup;
  // back up the state
  backup.rigid_ic = rigid_ic_;
  backup.bprior = bprior_;
  backup.errprior = errprior_;
  backup.depths.reserve(delta_x.rows() - pose_dim);
  for (int i = 0; i < feature::WINDOW_SIZE + 1; i++) {
    backup.Ps[i] = Ps[i];
    backup.Rs[i] = Rs[i];
    backup.Vs[i] = Vs[i];
    backup.Bas[i] = Bas[i];
    backup.Bgs[i] = Bgs[i];
  }

  // update extrinsic
  if (ESTIMATE_EXTRINSIC) {
    const Eigen::VectorXd& delta_pose = delta_x.segment(0, 6);
    if (isfinite(delta_pose.squaredNorm())) {
      rigid_ic_.translation() += delta_pose.head<3>();
      rigid_ic_.so3() = rigid_ic_.so3() * Sophus::SO3d::exp(delta_pose.tail<3>());
    }
  }

  // update pose
  for (int i = 0; i < feature::WINDOW_SIZE + 1; i++) {
    int idx = 6 + 15 * i;
    const Eigen::VectorXd& delta_pose = delta_x.segment(idx, 6);
    if (isfinite(delta_pose.squaredNorm())) {
      Ps[i] += delta_pose.head<3>();
      Rs[i] = (Sophus::SO3d(Eigen::Quaterniond(Rs[i])) * Sophus::SO3d::exp(delta_pose.tail<3>()))
                  .matrix();
    }

    const Eigen::VectorXd& delta_sb = delta_x.segment(idx + 6, 9);
    if (isfinite(delta_sb.squaredNorm())) {
      Vs[i] += delta_sb.segment(0, 3);
      Bas[i] += delta_sb.segment(3, 3);
      Bgs[i] += delta_sb.segment(6, 3);
    }
  }
  // update inverse depth
  int idx = pose_dim;
  for (auto& it_per_id : f_manager.feature) {
    if (!it_per_id.Valid()) continue;
    backup.depths.emplace_back(it_per_id.estimated_depth);
    double delta = delta_x(idx++);

    if (!isfinite(delta) || delta > 1e7) continue;
    double inv_depth = 1.0 / it_per_id.estimated_depth;
    it_per_id.estimated_depth = 1.0 / (inv_depth + delta);
  }

  // update prior
  if (valid_prior_) {
    /// update with first order Taylor, b' = b + \frac{\delta b}{\delta x} * \delta x
    /// \delta x = Computes the linearized deviation from the references (linearization points)
    bprior_ -= Hprior_ * delta_x.head(pose_dim);
    errprior_ = -Jprior_inv_ * bprior_.head(pose_dim - 15);
  }

  return backup;
}

Estimator::ProblemMeta Estimator::MakeProblem(int landmark_size) {
  static int pose_dim = (feature::WINDOW_SIZE + 1) * 15 + 6;
  int problem_size = pose_dim + landmark_size;
  Eigen::MatrixXd h_matrix(Eigen::MatrixXd::Zero(problem_size, problem_size));
  Eigen::VectorXd b_vec(Eigen::VectorXd::Zero(problem_size));
  double total_lost = 0;

  // add imu factors
  for (int i = 0; i < feature::WINDOW_SIZE; i++) {
    int j = i + 1;
    int idx_i = 6 + 15 * i;
    if (pre_integrations[j]->sum_dt > 10.0) continue;

    Eigen::VectorXd residual;
    std::vector<Eigen::MatrixXd> jacobians(4);
    std::vector<int> indices{idx_i, idx_i + 6, idx_i + 15, idx_i + 21};
    std::vector<int> dimensions{6, 9, 6, 9};
    backend::ComputeImuJacobian(Ps[i], Eigen::Quaterniond(Rs[i]), Vs[i], Bas[i], Bgs[i], Ps[j],
                                Eigen::Quaterniond(Rs[j]), Vs[j], Bas[j], Bgs[j],
                                pre_integrations[j], &(jacobians[0]), &(jacobians[1]),
                                &(jacobians[2]), &(jacobians[3]), &residual);
    Eigen::MatrixXd information = pre_integrations[j]->covariance.inverse();
    UpdateProblemMatrix(jacobians, indices, dimensions, residual, information, information, 1.0,
                        &h_matrix, &b_vec);
    total_lost += residual.transpose() * information * residual;
  }

  // add visual factors
  // make the problem hessian
  Eigen::MatrixXd vis_information = project_sqrt_info_.transpose() * project_sqrt_info_;

  int idx_feature = pose_dim - 1;

  Eigen::VectorXd residual;
  std::vector<Eigen::MatrixXd> jacobians(4);
  std::vector<int> dimensions{6, 6, ESTIMATE_EXTRINSIC ? 6 : -1, 1};
  // for all the features
  for (auto& it_per_id : f_manager.feature) {
    if (!it_per_id.Valid()) continue;

    int imu_i = it_per_id.start_frame;
    int idx_i = imu_i * 15 + 6;
    const Eigen::Vector3d& pt_1 = it_per_id.feature_per_frame[0].point;
    double inv_depth_i = 1.0 / it_per_id.estimated_depth;

    idx_feature++;
    if (inv_depth_i < 0) {
      continue;
    }

    if (it_per_id.depth_gt.has_value()) {
      // add inverse depth ground truth value factor
      double inv_depth_gt = 1.0 / it_per_id.depth_gt.value();
      double residual = inv_depth_i - inv_depth_gt;
      // jacobian is identity
      h_matrix(idx_feature, idx_feature) += depth_weight_;
      b_vec(idx_feature) -= depth_weight_ * residual;
    }

    for (size_t i = 1; i < it_per_id.feature_per_frame.size(); i++) {
      const Eigen::Vector3d& pt_2 = it_per_id.feature_per_frame[i].point;
      int imu_j = imu_i + i;
      int idx_j = imu_j * 15 + 6;
      std::vector<int> indices{idx_i, idx_j, 0, idx_feature};
      backend::ComputeVisualJacobian(
          rigid_ic_.translation(), rigid_ic_.so3().matrix(), Ps[imu_i], Rs[imu_i], Ps[imu_j],
          Rs[imu_j], inv_depth_i, pt_1, pt_2, &(jacobians[0]), &(jacobians[1]),
          ESTIMATE_EXTRINSIC ? &(jacobians[2]) : nullptr, &(jacobians[3]), &residual);

      // update robust loss fcn
      double drho;
      Eigen::MatrixXd robust_info;
      total_lost += backend::RobustInformation(*loss_fcn_.get(), vis_information,
                                               project_sqrt_info_, residual, &drho, &robust_info);

      // add to the final hessian matrix
      UpdateProblemMatrix(jacobians, indices, dimensions, residual, vis_information, robust_info,
                          drho, &h_matrix, &b_vec);
    }
  }

  // add prior
  int prior_dim = Hprior_.cols();
  if (valid_prior_) {
    h_matrix.topLeftCorner(prior_dim, prior_dim) += Hprior_;
    b_vec.head(prior_dim) += bprior_;
  }

  if (errprior_.rows() > 0) total_lost += errprior_.norm();
  total_lost *= 0.5;

  return Estimator::ProblemMeta{total_lost, h_matrix, b_vec};
}

bool SolveProblem(const Estimator::ProblemMeta& problem_meta, double lambda,
                  Eigen::VectorXd* delta_x) {
  static int pose_dim = (feature::WINDOW_SIZE + 1) * 15 + 6;
  int marg_size = problem_meta.h_matrix.cols() - pose_dim;
  const Eigen::MatrixXd& Hmm =
      problem_meta.h_matrix.block(pose_dim, pose_dim, marg_size, marg_size);
  const Eigen::MatrixXd& Hpm = problem_meta.h_matrix.block(0, pose_dim, pose_dim, marg_size);
  const Eigen::MatrixXd& Hmp = problem_meta.h_matrix.block(pose_dim, 0, marg_size, pose_dim);
  const Eigen::VectorXd& bpp = problem_meta.b_vec.segment(0, pose_dim);
  const Eigen::VectorXd& bmm = problem_meta.b_vec.segment(pose_dim, marg_size);

  Eigen::SparseMatrix<double> Hmm_inv(marg_size, marg_size);
  Hmm_inv.reserve(Eigen::VectorXi::Constant(marg_size, 1));
  for (int i = 0; i < marg_size; i++) {
    Hmm_inv.insert(i, i) = 1.0 / Hmm(i, i);
  }
  Hmm_inv.makeCompressed();

  Eigen::MatrixXd tempH = Hpm * Hmm_inv;
  Eigen::MatrixXd H_pp_schur = problem_meta.h_matrix.block(0, 0, pose_dim, pose_dim) - tempH * Hmp;
  Eigen::VectorXd b_pp_schur = bpp - tempH * bmm;

  for (int i = 0; i < pose_dim; i++) {
    H_pp_schur(i, i) += lambda;
  }
  // solve Hpp * delta_x = bpp
  Eigen::VectorXd delta_x_pp = H_pp_schur.ldlt().solve(b_pp_schur);

  // solve Hmm * delta_x = bmm - Hmp * delta_x_pp;
  Eigen::VectorXd delta_x_ll = Hmm_inv * (bmm - Hmp * delta_x_pp);

  *delta_x = Eigen::VectorXd(problem_meta.h_matrix.cols());
  delta_x->head(pose_dim) = delta_x_pp;
  delta_x->tail(marg_size) = delta_x_ll;

  return isfinite(delta_x->squaredNorm());
}

double ComputeInitialLambda(const Estimator::ProblemMeta& problem) {
  double max_diagonal = 0;
  for (int i = 0; i < problem.h_matrix.cols(); ++i) {
    max_diagonal = std::max(fabs(problem.h_matrix(i, i)), max_diagonal);
  }
  max_diagonal = std::min(5e10, max_diagonal);
  return 1e-5 * max_diagonal;
}

bool IsGoodStep(const Estimator::ProblemMeta& problem, const Eigen::VectorXd& delta_x,
                double current_loss, double last_lost, double* lambda, double* ni) {
  double scale = 0.5 * delta_x.transpose() * ((*lambda) * delta_x + problem.b_vec) + 1e-6;
  double rho = (last_lost - current_loss) / scale;
  if (rho > 0 && isfinite(current_loss)) {
    // last step was good
    double alpha = 1. - std::pow((2 * rho - 1), 3);
    alpha = std::min(alpha, 2. / 3.);
    double scale_factor = (std::max)(1. / 3., alpha);
    *lambda *= scale_factor;
    *ni = 2;
    return true;
  } else {
    *lambda *= *ni;
    *ni *= 2;
    return false;
  }
}

double Estimator::OptimizeSlideWindow(int max_num_iterations, bool veb) {
  TicToc t_solve;

  int landmark_size = 0;
  int pt_has_gt = 0;
  for (auto& it_per_id : f_manager.feature) {
    if (!it_per_id.Valid()) continue;
    landmark_size++;
    pt_has_gt += (it_per_id.depth_gt.has_value());
  }

  LOG_IF(INFO, verbose_) << "landmark points : " << pt_has_gt << "/" << landmark_size;

  ProblemMeta problem = MakeProblem(landmark_size);
  double lambda = ComputeInitialLambda(problem);
  double ni = 2.0;
  double last_cost = problem.total_lost;
  std::vector<double> sqr_residuals;

  if (verbose_ && veb) {
    std::printf(" | %10s | %13s | %13s | %13s |\n", "Iteration", "Residual", "Norm Dx", "Lambda");
    std::printf(" | %10d | %13.6f | %13.6f | %13.6f |\n", -1, last_cost, 0., lambda);
  }
  for (int iter = 0; iter < max_num_iterations; iter++) {
    Eigen::VectorXd delta_x;
    for (int lm_try = 0; lm_try < 8; lm_try++) {
      if (!SolveProblem(problem, lambda, &delta_x)) {
        ComputeInitialLambda(problem);
        continue;
      }
      // SolveProblem(problem, lambda, &delta_x);
      States backup_state = UpdateStates(delta_x);
      double current_loss = ComputeCurrentLoss(&sqr_residuals);
      if (IsGoodStep(problem, delta_x, current_loss, last_cost, &lambda, &ni)) {
        break;
      }
      // it was a bad update roll back to previous state
      RollbackStates(backup_state);
    }

    problem = MakeProblem(landmark_size);
    double dx_norm = delta_x.norm();

    if (verbose_ && veb) {
      std::printf(" | %10d | %13.6f | %13.6f | %13.6f |\n", iter, problem.total_lost, dx_norm,
                  lambda);
    }
    if (last_cost - problem.total_lost < 1e-6) {
      LOG_IF(INFO, verbose_) << " [LM STOP] the residual reduce too less (< 1e-6)";
      break;
    }
    last_cost = problem.total_lost;
  }
  LOG_IF(INFO, verbose_) << " [LM FINISH]  problem solve cost: " << t_solve.toc() << " ms";

  return 1e-4;
  CHECK(!sqr_residuals.empty());
  std::sort(sqr_residuals.begin(), sqr_residuals.end());
  return sqr_residuals[0.9 * sqr_residuals.size()];
}

int Estimator::BackendOptimization() {
  Eigen::Matrix3d rot_0 = Rs[0];
  Eigen::Vector3d pos_0 = Ps[0];

  double sqr_residual = OptimizeSlideWindow(NUM_ITERATIONS, false);

  // the optimization may move the first frame a lot
  // move the first frame towards its origin position
  {
    Eigen::Vector3d pos_0_new = Ps[0];

    // as the optimization may change the pose of all the frames (even though we have prior)
    // if the first frames pose changed, this will lead to the system random walk behaviour
    // to solve this, we calculate the difference between the first frame pose before and
    // after the optimization, then propragate it to all other window frames
    //      the system has one rotation DOF and three poisition DOF
    Eigen::Matrix3d rot_diff = Eigen::Matrix3d::Identity();
    if (false) {
      Eigen::Vector3d euler_0 = backend::Utility::R2ypr(rot_0);
      Eigen::Vector3d euler_0_new = backend::Utility::R2ypr(Rs[0]);
      double y_diff = euler_0.x() - euler_0_new.x();
      rot_diff = backend::Utility::ypr2R(Vector3d(y_diff, 0, 0));
      if (abs(abs(euler_0.y()) - 90.0) < 1.0 || abs(abs(euler_0_new.y()) - 90.0) < 1.0) {
        rot_diff = rot_0 * Rs[0].transpose();
      }
    }
    for (int i = 0; i <= feature::WINDOW_SIZE + 1; i++) {
      Rs[i] = rot_diff * Rs[i];
      Ps[i] = rot_diff * (Ps[i] - pos_0_new) + pos_0;
      Vs[i] = rot_diff * Vs[i];
    }
  }

  if (marginalization_flag == MARGIN_OLD) {
    MargOldFrame();
  } else if (valid_prior_) {
    MargNewFrame();
  }

  // repropagate
  // for (int i = 1; i <= feature::WINDOW_SIZE; i++) {
  //   pre_integrations[i]->repropagate(Bas[i], Bgs[i - 1]);
  // }
  return 100;

  double threshold = std::max(1e-3, sqr_residual);
  auto outlier_inlier = RejectOutliers(threshold);
  LOG_IF(INFO, verbose_) << "reject outlier " << outlier_inlier.first << " with threshold " << threshold;
  return outlier_inlier.second;
}

}  // namespace vins
