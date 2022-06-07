
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

Estimator::States Estimator::UpdateStates(const Eigen::VectorXd& delta_x) {
  static int pose_dim = (feature::WINDOW_SIZE + 1) * 15 + 6;
  States backup;
  // back up the state
  backup.rigid_ic = rigid_ic_;
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
    rigid_ic_.translation() += delta_pose.head<3>();
    rigid_ic_.so3() = rigid_ic_.so3() * Sophus::SO3d::exp(delta_pose.tail<3>());
  }

  // update pose
  for (int i = 0; i < feature::WINDOW_SIZE + 1; i++) {
    int idx = 6 + 15 * i;
    const Eigen::VectorXd& delta_pose = delta_x.segment(idx, 6);
    Ps[i] += delta_pose.head<3>();
    Rs[i] = Rs[i] * Sophus::SO3d::exp(delta_pose.tail<3>()).matrix();

    const Eigen::VectorXd& delta_sb = delta_x.segment(idx + 6, 9);
    Vs[i] += delta_sb.segment(0, 3);
    Bas[i] += delta_sb.segment(3, 3);
    Bgs[i] += delta_sb.segment(6, 3);
  }

  // update inverse depth
  int idx = pose_dim;
  for (auto& it_per_id : f_manager.feature) {
    if (!it_per_id.Valid()) continue;

    double inv_depth = 1.0 / it_per_id.estimated_depth;
    it_per_id.estimated_depth = 1.0 / (inv_depth + delta_x(idx++));
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

  int idx_feature = 0;

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
  h_matrix.topLeftCorner(prior_dim, prior_dim) += Hprior_;
  b_vec.head(prior_dim) += bprior_;

  // update lm parameter
  if (errprior_.rows() > 0) total_lost += errprior_.norm();
  total_lost *= 0.5;

  return Estimator::ProblemMeta{total_lost, h_matrix, b_vec};
}

bool Estimator::SolveProblem(const Estimator::ProblemMeta& problem_meta, double lambda, Eigen::VectorXd* delta_x) {
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

  // solve failed, as the delta x is much too large
  return delta_x->squaredNorm() < 1e6;
}

void Estimator::ProblemSolve() {
  vInverseDepth = f_manager.GetInverseDepthVector();

  // step1. build the problem
  backend::Problem problem(backend::Problem::ProblemType::SLAM_PROBLEM);
  vector<shared_ptr<backend::VertexPose>> vertexCams_vec;
  vector<shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
  int pose_dim = 0;

  // add the externion parameters to the graph, body camera transformation, camera calibrations,
  // etc. as it is frequency used, put it in the first place.
  shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
  {
    Eigen::VectorXd pose = vPic[0];
    vertexExt->SetParameters(pose);
    if (!ESTIMATE_EXTRINSIC) {
      vertexExt->SetFixed();
    }
    problem.AddVertex(vertexExt);
    pose_dim += vertexExt->LocalDimension();
  }

  for (int i = 0; i < feature::WINDOW_SIZE + 1; i++) {
    shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
    Quaterniond q_init(Rs[i]);
    Eigen::VectorXd pose(7);
    pose << Ps[i][0], Ps[i][1], Ps[i][2], q_init.x(), q_init.y(), q_init.z(), q_init.w();
    vertexCam->SetParameters(pose);
    vertexCams_vec.push_back(vertexCam);
    problem.AddVertex(vertexCam);
    pose_dim += vertexCam->LocalDimension();

    shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
    Eigen::VectorXd vb(9);
    vb << Vs[i][0], Vs[i][1], Vs[i][2], Bas[i][0], Bas[i][1], Bas[i][2], Bgs[i][0], Bgs[i][1],
        Bgs[i][2];
    vertexVB->SetParameters(vb);
    vertexVB_vec.push_back(vertexVB);
    problem.AddVertex(vertexVB);
    pose_dim += vertexVB->LocalDimension();
  }

  // IMU
  for (int i = 0; i < feature::WINDOW_SIZE; i++) {
    int j = i + 1;
    if (pre_integrations[j]->sum_dt > 10.0) continue;

    std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
    edge_vertex.push_back(vertexCams_vec[i]);
    edge_vertex.push_back(vertexVB_vec[i]);
    edge_vertex.push_back(vertexCams_vec[j]);
    edge_vertex.push_back(vertexVB_vec[j]);

    std::shared_ptr<backend::EdgeImu> imuEdge =
        std::make_shared<backend::EdgeImu>(pre_integrations[j], edge_vertex);
    problem.AddEdge(imuEdge);
  }

  // Visual Factor
  std::vector<std::shared_ptr<backend::VertexInverseDepth>> vertexPt_vec;
  int feature_index = 0;
  // for all the features
  for (auto& it_per_id : f_manager.feature) {
    if (it_per_id.feature_per_frame.size() < 2) continue;
    if (it_per_id.start_frame > feature::WINDOW_SIZE - 3) continue;

    int imu_i = it_per_id.start_frame;
    const Eigen::Vector3d& pts_i = it_per_id.feature_per_frame[0].point;

    std::shared_ptr<backend::VertexInverseDepth> verterxPoint(new backend::VertexInverseDepth());
    verterxPoint->SetParameters(Eigen::Matrix<double, 1, 1>(vInverseDepth[feature_index++]));
    problem.AddVertex(verterxPoint);
    vertexPt_vec.push_back(verterxPoint);

    for (size_t i = 1; i < it_per_id.feature_per_frame.size(); i++) {
      const Eigen::Vector3d& pts_j = it_per_id.feature_per_frame[i].point;

      std::vector<std::shared_ptr<backend::Vertex>> edge_vertex{
          verterxPoint, vertexCams_vec[imu_i], vertexCams_vec[imu_i + i], vertexExt};

      std::shared_ptr<backend::EdgeReprojection> edge =
          std::make_shared<backend::EdgeReprojection>(pts_i, pts_j, edge_vertex);
      edge->SetInformation(project_sqrt_info_.transpose() * project_sqrt_info_);
      edge->SetLossFunction(loss_fcn_);
      problem.AddEdge(edge);
    }
  }

  // prior process already got one
  if (Hprior_.rows() > 0) {
    problem.SetHessianPrior(Hprior_);  // tell the problem
    problem.SetbPrior(bprior_);
    problem.SetErrPrior(errprior_);
    problem.SetJtPrior(Jprior_inv_);
    problem.ExtendHessiansPriorSize(15);  // extand the hessian prior
  }

  // LOG(INFO) << "current residual : " << ComputeCurrentLoss();
  problem.Solve(NUM_ITERATIONS, false);

  // update bprior_,  Hprior_ do not need update
  if (Hprior_.rows() > 0) {
    bprior_ = problem.GetbPrior();
    errprior_ = problem.GetErrPrior();
  }

  // BASTIAN_M : directly update the vectors, instead of call double 2 vector later.
  // for the optimized variables : double2vector() of the original project
  Eigen::Vector3d origin_R0 = backend::Utility::R2ypr(Rs[0]);
  Eigen::Vector3d origin_P0 = Ps[0];

  VecX vPoseCam0 = vertexCams_vec[0]->Parameters();
  Eigen::Matrix3d mRotCam0 =
      Quaterniond(vPoseCam0[6], vPoseCam0[3], vPoseCam0[4], vPoseCam0[5]).toRotationMatrix();
  Vector3d origin_R00 = backend::Utility::R2ypr(mRotCam0);
  double y_diff = origin_R0.x() - origin_R00.x();
  // LOG(INFO) << y_diff;

  // as the optimization may change the pose of all the frames
  // if the first frames pose changed, this will lead to the system random walk behaviour
  // to solve this, we calculate the difference between the frist frame pose before and
  // after the optimization, then propragate it to all other window frames
  //      the system has one rotation DOF and three poisition DOF
  Matrix3d rot_diff = backend::Utility::ypr2R(Vector3d(y_diff, 0, 0));
  if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0) {
    rot_diff = Rs[0] * mRotCam0.transpose();
  }

  // rot_diff = Matrix3d::Identity();
  for (int i = 0; i <= feature::WINDOW_SIZE; i++) {
    VecX vPoseCam_i = vertexCams_vec[i]->Parameters();
    Rs[i] = rot_diff * Quaterniond(vPoseCam_i[6], vPoseCam_i[3], vPoseCam_i[4], vPoseCam_i[5])
                           .normalized()
                           .toRotationMatrix();

    Ps[i] = rot_diff * Vector3d(vPoseCam_i[0] - vPoseCam0[0], vPoseCam_i[1] - vPoseCam0[1],
                                vPoseCam_i[2] - vPoseCam0[2]) +
            origin_P0;

    VecX vSpeedBias_i = vertexVB_vec[i]->Parameters();
    Vs[i] = rot_diff * Vector3d(vSpeedBias_i[0], vSpeedBias_i[1], vSpeedBias_i[2]);
    Bas[i] = Vector3d(vSpeedBias_i[3], vSpeedBias_i[4], vSpeedBias_i[5]);
    Bgs[i] = Vector3d(vSpeedBias_i[6], vSpeedBias_i[7], vSpeedBias_i[8]);
  }

  if (ESTIMATE_EXTRINSIC) {
    VecX vExterCali = vertexExt->Parameters();
    rigid_ic_ =
        Sophus::SE3d(Eigen::Quaterniond(vExterCali[6], vExterCali[3], vExterCali[4], vExterCali[5]),
                     Eigen::Vector3d(vExterCali[0], vExterCali[1], vExterCali[2]));
    vPic[0] << vExterCali[0], vExterCali[1], vExterCali[2], vExterCali[3], vExterCali[4],
        vExterCali[5], vExterCali[6];
  }

  int f_count = f_manager.GetFeatureCount();
  VectorXd vInvDepToSet(f_count);
  for (int i = 0; i < f_count; ++i) {
    VecX f = vertexPt_vec[i]->Parameters();
    vInvDepToSet(i) = f[0];
    vInverseDepth(i) = f[0];
  }
  f_manager.SetDepth(vInvDepToSet);
}

void Estimator::BackendOptimization() {
  ProblemSolve();

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
