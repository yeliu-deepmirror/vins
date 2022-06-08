
#include "vins/estimator.h"

namespace vins {

void UpdateProblemMatrix(const std::vector<Eigen::MatrixXd>& jacobians,
                         const std::vector<int>& indices, const std::vector<int>& dimensions,
                         const Eigen::VectorXd& residual, const Eigen::MatrixXd& information,
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
    b_vec->segment(indices[a], dimensions[a]) -=
        drho * jacobians[a].transpose() * information * residual;
  }
}

void Estimator::MargOldFrame() {
  int pose_dim = (feature::WINDOW_SIZE + 1) * 15 + 6;

  // make the marginalization matrix from the edges (only visual edges)
  if (valid_prior_) {
    ExtendedPrior(15);
  } else {
    Hprior_ = Eigen::MatrixXd::Zero(pose_dim, pose_dim);
    bprior_ = Eigen::VectorXd::Zero(pose_dim);
    valid_prior_ = true;
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
    UpdateProblemMatrix(jacobians, indices, dimensions, residual, information, information, 1.0,
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

    if (it_per_id.depth_gt.has_value()) {
      // add inverse depth ground truth value factor
      double inv_depth_gt = 1.0 / it_per_id.depth_gt.value();
      double residual = inv_depth_i - inv_depth_gt;
      // jacobian is identity
      marginalization_matrix(idx_feature, idx_feature) += depth_weight_;
      marginalization_vec(idx_feature) -= depth_weight_ * residual;
    }

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
      UpdateProblemMatrix(jacobians, indices, dimensions, residual, information, robust_info, drho,
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
  CHECK(valid_prior_);
  ExtendedPrior(15);

  backend::MarginalizeFrameInternal(6 + (feature::WINDOW_SIZE - 1) * 15, 15, &Hprior_, &bprior_,
                                    &errprior_, &Jprior_inv_);
  if (!ESTIMATE_EXTRINSIC) {
    Hprior_.block(0, 0, 6, Hprior_.cols()).setZero();
    Hprior_.block(0, 0, Hprior_.rows(), 6).setZero();
    bprior_.segment(0, 6).setZero();
  }
}

}  // namespace vins
