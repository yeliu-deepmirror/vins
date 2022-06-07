
#include "vins/estimator.h"

namespace vins {
namespace {

inline void UpdateProblemMatrix(const std::vector<Eigen::MatrixXd>& jacobians,
                                const std::vector<int>& indices, const std::vector<int>& dimensions,
                                const Eigen::VectorXd& residual,
                                const Eigen::MatrixXd& robust_information, double drho,
                                Eigen::MatrixXd* h_mat, Eigen::VectorXd* b_vec) {
  // add to the final hessian matrix
  for (size_t a = 0; a < jacobians.size(); a++) {
    if (dimensions[a] < 0) {
      // we use negative dimension as indicator for fixed node
      continue;
    }
    Eigen::MatrixXd jtw = jacobians[a].transpose() * robust_information;
    for (size_t b = a; b < jacobians.size(); b++) {
      if (dimensions[b] < 0) {
        // we use negative dimension as indicator for fixed node
        continue;
      }
      Eigen::MatrixXd hessian = jtw * jacobians[b];
      h_mat->block(indices[a], indices[b], dimensions[a], dimensions[b]) += hessian;
      if (a != b) {
        h_mat->block(indices[b], indices[a], dimensions[b], dimensions[a]) += hessian.transpose();
      }
    }
    b_vec->segment(indices[a], dimensions[a]) -= drho * jtw * residual;
  }
}
}  // namespace

void Estimator::MargOldFrame() {
  int pose_dim = (feature::WINDOW_SIZE + 1) * 15 + 6;

  // make the marginalization matrix from the edges (only visual edges)
  if (Hprior_.rows() > 0) {
    ExtendedPrior(15);
  } else {
    Hprior_ = Eigen::MatrixXd::Zero(pose_dim, pose_dim);
    bprior_ = Eigen::VectorXd::Zero(pose_dim);
  }

  Eigen::MatrixXd information = project_sqrt_info_.transpose() * project_sqrt_info_;

  // compute marginalization observation size
  int marg_landmark_size = 0;
  for (auto& it_per_id : f_manager.feature) {
    if (!it_per_id.Valid()) continue;
    if (it_per_id.start_frame != 0) continue;
    marg_landmark_size++;
  }

  int marg_size = pose_dim + marg_landmark_size;
  Eigen::MatrixXd marginalization_matrix(Eigen::MatrixXd::Zero(marg_size, marg_size));
  Eigen::VectorXd marginalization_vec(Eigen::VectorXd::Zero(marg_size));

  // for the imu factor
  if (pre_integrations[1]->sum_dt < 10.0) {
    Eigen::VectorXd residual;
    std::vector<Eigen::MatrixXd> jacobians(4);
    std::vector<int> indices{6, 12, 21, 27};
    std::vector<int> dimensions{6, 9, 6, 9};
    backend::ComputeImuJacobian(Ps[0], Eigen::Quaterniond(Rs[0]), Vs[0], Bas[0], Bgs[0], Ps[1],
                                Eigen::Quaterniond(Rs[1]), Vs[1], Bas[1], Bgs[1],
                                pre_integrations[1], &(jacobians[0]), &(jacobians[1]),
                                &(jacobians[2]), &(jacobians[3]), &residual);

    Eigen::MatrixXd information = pre_integrations[1]->covariance.inverse();
    UpdateProblemMatrix(jacobians, indices, dimensions, residual, information, 1.0,
                        &marginalization_matrix, &marginalization_vec);
  }

  int landmark_id = 0;
  int feature_index = -1;
  // for all the viewed features
  for (auto& it_per_id : f_manager.feature) {
    if (!it_per_id.Valid()) continue;
    ++feature_index;
    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
    if (imu_i != 0) continue;

    const Eigen::Vector3d& pt_1 = it_per_id.feature_per_frame[0].point;
    double inv_depth_i = 1.0 / it_per_id.estimated_depth;
    int idx_i = imu_i * 15 + 6;
    int idx_feature = pose_dim + landmark_id++;

    {
      // add point to map
      Eigen::Vector3d pt_camera_1 = pt_1 / inv_depth_i;
      all_map_points_[it_per_id.feature_id] = Rs[imu_i] * (rigid_ic_ * pt_camera_1) + Ps[imu_i];
    }

    // for all its observations -> each has a reprojection error w.r.t the first observation
    for (auto& it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      if (imu_i == imu_j) continue;
      int idx_j = imu_j * 15 + 6;
      const Eigen::Vector3d& pt_2 = it_per_frame.point;

      Eigen::VectorXd residual;
      std::vector<Eigen::MatrixXd> jacobians(4);
      std::vector<int> indices{idx_i, idx_j, 0, idx_feature};
      std::vector<int> dimensions{6, 6, ESTIMATE_EXTRINSIC ? 6 : -1, 1};
      backend::ComputeVisualJacobian(
          rigid_ic_.translation(), rigid_ic_.so3().matrix(), Ps[imu_i], Rs[imu_i], Ps[imu_j],
          Rs[imu_j], inv_depth_i, pt_1, pt_2, &(jacobians[0]), &(jacobians[1]),
          ESTIMATE_EXTRINSIC ? &(jacobians[2]) : nullptr, &(jacobians[3]), &residual);

      // update robust loss fcn
      double drho;
      Eigen::MatrixXd robust_info;
      backend::RobustInformation(*loss_fcn_.get(), information, project_sqrt_info_, residual, &drho,
                                 &robust_info);

      // add to the final hessian matrix
      UpdateProblemMatrix(jacobians, indices, dimensions, residual, robust_info, drho,
                          &marginalization_matrix, &marginalization_vec);
    }
  }
  CHECK_EQ(landmark_id, marg_landmark_size);

  // solve the marginalization problem
  {
    MatXX Hmm =
        marginalization_matrix.block(pose_dim, pose_dim, marg_landmark_size, marg_landmark_size);
    MatXX Hpm = marginalization_matrix.block(0, pose_dim, pose_dim, marg_landmark_size);
    MatXX Hmp = marginalization_matrix.block(pose_dim, 0, marg_landmark_size, pose_dim);
    VecX bpp = marginalization_vec.segment(0, pose_dim);
    VecX bmm = marginalization_vec.segment(pose_dim, marg_landmark_size);

    // use sparse matrix to accelerate and save memory
    // tutorial : https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
    // as for vins system, we use inverse depth, each vertex will only have one value
    // we will reserve the space for memory savage
    Eigen::SparseMatrix<double> Hmm_inv(marg_landmark_size, marg_landmark_size);
    Hmm_inv.reserve(Eigen::VectorXi::Constant(marg_landmark_size, 1));
    for (int i = 0; i < marg_landmark_size; i++) {
      Hmm_inv.insert(i, i) = 1.0 / Hmm(i, i);
    }
    Hmm_inv.makeCompressed();

    MatXX tempH = Hpm * Hmm_inv;
    Hprior_ = marginalization_matrix.block(0, 0, pose_dim, pose_dim) - tempH * Hmp + Hprior_;
    bprior_ = bpp - tempH * bmm + bprior_;
  }

  backend::MarginalizeFrameInternal(6, 15, &Hprior_, &bprior_, &errprior_, &Jprior_inv_);

  if (!ESTIMATE_EXTRINSIC) {
    Hprior_.block(0, 0, 6, Hprior_.cols()).setZero();
    Hprior_.block(0, 0, Hprior_.rows(), 6).setZero();
    bprior_.segment(0, 6).setZero();
  }
}

void Estimator::ExtendedPrior(int dim) {
  int size = Hprior_.rows() + dim;
  Hprior_.conservativeResize(size, size);
  bprior_.conservativeResize(size);
  bprior_.tail(dim).setZero();
  Hprior_.rightCols(dim).setZero();
  Hprior_.bottomRows(dim).setZero();
}

void Estimator::MargNewFrame() {
  // if we marginalize the new frame, no map point observation shall be marginalized.
  if (Hprior_.rows() > 0) {
    ExtendedPrior(15);
  } else {
    int pose_dim = feature::WINDOW_SIZE * 15 + 6;
    Hprior_ = MatXX(pose_dim, pose_dim);
    Hprior_.setZero();
    bprior_ = VecX(pose_dim);
    bprior_.setZero();
  }
  backend::MarginalizeFrameInternal(6 + (feature::WINDOW_SIZE - 1) * 15, 15, &Hprior_, &bprior_,
                                    &errprior_, &Jprior_inv_);
  if (!ESTIMATE_EXTRINSIC) {
    Hprior_.block(0, 0, 6, Hprior_.cols()).setZero();
    Hprior_.block(0, 0, Hprior_.rows(), 6).setZero();
    bprior_.segment(0, 6).setZero();
  }
}

// TODO(yeliu) : refine using :
// github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/trust_region_minimizer.cc#L415
double Estimator::ComputeCurrentLoss() {
  double total_lost = 0;
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

    for (size_t i = 1; i < it_per_id.feature_per_frame.size(); i++) {
      const Eigen::Vector3d& pt_2 = it_per_id.feature_per_frame[i].point;
      int imu_j = imu_i + i;
      Eigen::VectorXd residual = backend::ComputeVisualResidual(
          rigid_ic_.translation(), rigid_ic_.so3().matrix(), Ps[imu_i], Rs[imu_i], Ps[imu_j],
          Rs[imu_j], inv_depth_i, pt_1, pt_2);

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

    if (!isfinite(delta) || delta > 1e3) continue;
    double inv_depth = 1.0 / it_per_id.estimated_depth;
    it_per_id.estimated_depth = 1.0 / (inv_depth + delta);
  }

  // update prior
  if (errprior_.rows() > 0) {
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
    UpdateProblemMatrix(jacobians, indices, dimensions, residual, information, 1.0, &h_matrix,
                        &b_vec);
    total_lost += residual.transpose() * information * residual;
  }

  // add visual factors
  // make the problem hessian
  Eigen::MatrixXd vis_information = project_sqrt_info_.transpose() * project_sqrt_info_;

  int idx_feature = pose_dim;

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
      UpdateProblemMatrix(jacobians, indices, dimensions, residual, robust_info, drho, &h_matrix,
                          &b_vec);
    }
    idx_feature++;
  }

  // add prior
  int prior_dim = Hprior_.cols();
  if (prior_dim > 0) {
    h_matrix.topLeftCorner(prior_dim, prior_dim) += Hprior_;
    b_vec.head(prior_dim) += bprior_;
  }

  // update lm parameter
  if (errprior_.rows() > 0) total_lost += errprior_.norm();
  total_lost *= 0.5;

  return Estimator::ProblemMeta{total_lost, h_matrix, b_vec};
}

bool Estimator::SolveProblem(const Estimator::ProblemMeta& problem_meta, double lambda,
                             Eigen::VectorXd* delta_x) {
  static int pose_dim = (feature::WINDOW_SIZE + 1) * 15 + 6;
  int marg_size = problem_meta.h_matrix.cols() - pose_dim;
  Eigen::MatrixXd Hmm = problem_meta.h_matrix.block(pose_dim, pose_dim, marg_size, marg_size);
  Eigen::MatrixXd Hpm = problem_meta.h_matrix.block(0, pose_dim, pose_dim, marg_size);
  Eigen::MatrixXd Hmp = problem_meta.h_matrix.block(pose_dim, 0, marg_size, pose_dim);
  Eigen::VectorXd bpp = problem_meta.b_vec.segment(0, pose_dim);
  Eigen::VectorXd bmm = problem_meta.b_vec.segment(pose_dim, marg_size);

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
    H_pp_schur(i, i) += lambda;  // LM Method
  }
  // solve Hpp * delta_x = bpp
  Eigen::VectorXd delta_x_pp = H_pp_schur.ldlt().solve(b_pp_schur);

  // solve Hmm * delta_x = bmm - Hmp * delta_x_pp;
  Eigen::VectorXd delta_x_ll = Hmm_inv * (bmm - Hmp * delta_x_pp);

  *delta_x = Eigen::VectorXd(problem_meta.h_matrix.cols());
  delta_x->head(pose_dim) = delta_x_pp;
  delta_x->tail(marg_size) = delta_x_ll;

  // return isfinite(delta_x->squaredNorm());
  return true;
}

double Estimator::ComputeInitialLambda(const Estimator::ProblemMeta& problem) {
  double max_diagonal = 0;
  for (int i = 0; i < problem.h_matrix.cols(); ++i) {
    max_diagonal = std::max(fabs(problem.h_matrix(i, i)), max_diagonal);
  }
  max_diagonal = std::min(5e10, max_diagonal);
  return 1e-5 * max_diagonal;
}

bool Estimator::IsGoodStep(const Estimator::ProblemMeta& problem, const Eigen::VectorXd& delta_x,
                           double last_lost, double* lambda, double* ni) {
  double scale = 0.5 * delta_x.transpose() * ((*lambda) * delta_x + problem.b_vec) + 1e-6;
  double current_loss = ComputeCurrentLoss();

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

void Estimator::OptimizeSlideWindow(int max_num_iterations) {
  TicToc t_solve;

  int landmark_size = 0;
  for (auto& it_per_id : f_manager.feature) {
    if (!it_per_id.Valid()) continue;
    landmark_size++;
  }

  ProblemMeta problem = MakeProblem(landmark_size);
  double lambda = ComputeInitialLambda(problem);
  double ni = 2.0;
  double last_cost = problem.total_lost;

  if (verbose_) {
    std::printf(" | %10s | %13s | %13s | %13s |\n", "Iteration", "Residual", "Norm Dx", "Lambda");
    std::printf(" | %10d | %13.6f | %13.6f | %13.6f |\n", -1, last_cost, 0., lambda);
  }
  for (int iter = 0; iter < max_num_iterations; iter++) {
    Eigen::VectorXd delta_x;
    for (int lm_try = 0; lm_try < 8; lm_try++) {
      // if (!SolveProblem(problem, lambda, &delta_x)) {
      //   ComputeInitialLambda(problem);
      //   continue;
      // }
      SolveProblem(problem, lambda, &delta_x);
      States backup_state = UpdateStates(delta_x);
      if (IsGoodStep(problem, delta_x, last_cost, &lambda, &ni)) {
        break;
      }
      // it was a bad update roll back to previous state
      RollbackStates(backup_state);
    }

    problem = MakeProblem(landmark_size);
    double dx_norm = delta_x.norm();

    if (verbose_) {
      std::printf(" | %10d | %13.6f | %13.6f | %13.6f |\n", iter, problem.total_lost, dx_norm,
                  lambda);
    }

    // if the current residual improvement too small or maybe the linear system solved failed
    if (last_cost - problem.total_lost < 1e-6) {
      if (verbose_) {
        std::cout << " [LM STOP] the residual reduce too less (< 1e-6)" << std::endl;
      }
      break;
    }
    last_cost = problem.total_lost;
  }
  if (verbose_) {
    std::cout << " [LM FINISH]  problem solve cost: " << t_solve.toc() << " ms" << std::endl;
  }
}

void Estimator::BackendOptimization() {
  OptimizeSlideWindow(NUM_ITERATIONS);

  if (marginalization_flag == MARGIN_OLD) {
    MargOldFrame();
  } else {
    // if have prior
    if (Hprior_.rows() > 0) {
      MargNewFrame();
    }
  }
}

}  // namespace vins
