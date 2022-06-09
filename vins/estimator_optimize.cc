
#include "vins/estimator.h"

namespace vins {

void Estimator::ProblemSolve() {
  vInverseDepth = f_manager.GetInverseDepthVector();

  std::shared_ptr<backend::LossFunction> lossfunction = std::make_shared<backend::CauchyLoss>(1.0);

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
  {
    int feature_index = -1;
    // for all the features
    for (auto& it_per_id : f_manager.feature) {
      it_per_id.used_num = it_per_id.feature_per_frame.size();
      if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < feature::WINDOW_SIZE - 2)) continue;

      ++feature_index;

      int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
      Vector3d pts_i = it_per_id.feature_per_frame[0].point;

      shared_ptr<backend::VertexInverseDepth> verterxPoint(new backend::VertexInverseDepth());
      VecX inv_d(1);
      inv_d << vInverseDepth[feature_index];
      verterxPoint->SetParameters(inv_d);
      problem.AddVertex(verterxPoint);
      vertexPt_vec.push_back(verterxPoint);

      if (it_per_id.inv_depth_gt_.has_value()) {
        // if has depth ground truth add factor for it
        std::vector<std::shared_ptr<backend::Vertex>> edge_vertex{verterxPoint};
        std::shared_ptr<backend::EdgeInvDepthPrior> edge =
            std::make_shared<backend::EdgeInvDepthPrior>(it_per_id.inv_depth_gt_.value(), edge_vertex);
        edge->SetInformation(depth_information_);
        problem.AddEdge(edge);
      }

      // for all its observations
      for (auto& it_per_frame : it_per_id.feature_per_frame) {
        imu_j++;
        if (imu_i == imu_j) continue;

        Vector3d pts_j = it_per_frame.point;

        std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
        edge_vertex.push_back(verterxPoint);
        edge_vertex.push_back(vertexCams_vec[imu_i]);
        edge_vertex.push_back(vertexCams_vec[imu_j]);
        edge_vertex.push_back(vertexExt);

        std::shared_ptr<backend::EdgeReprojection> edge =
            std::make_shared<backend::EdgeReprojection>(pts_i, pts_j, edge_vertex);
        edge->SetInformation(project_sqrt_info_.transpose() * project_sqrt_info_);
        edge->SetLossFunction(lossfunction);
        problem.AddEdge(edge);
      }
    }
  }

  // prior process
  {
    // already got one
    if (Hprior_.rows() > 0) {
      problem.SetHessianPrior(Hprior_);  // tell the problem
      problem.SetbPrior(bprior_);
      problem.SetErrPrior(errprior_);
      problem.SetJtPrior(Jprior_inv_);
      problem.ExtendHessiansPriorSize(15);  // extand the hessian prior
    }
  }

  problem.Solve(NUM_ITERATIONS, verbose_);

  // update bprior_,  Hprior_ do not need update
  if (Hprior_.rows() > 0) {
    bprior_ = problem.GetbPrior();
    errprior_ = problem.GetErrPrior();
  }

  // BASTIAN_M : directly update the vectors, instead of call double 2 vector later.
  // for the optimized variables : double2vector() of the original project
  Eigen::Vector3d origin_R0 = backend::Utility::R2ypr(Rs[0]);
  Eigen::Vector3d origin_P0 = Ps[0];

  if (failure_occur) {
    origin_R0 = backend::Utility::R2ypr(last_R0);
    origin_P0 = last_P0;
    failure_occur = 0;
  }
  VecX vPoseCam0 = vertexCams_vec[0]->Parameters();
  Eigen::Matrix3d mRotCam0 =
      Quaterniond(vPoseCam0[6], vPoseCam0[3], vPoseCam0[4], vPoseCam0[5]).toRotationMatrix();
  Vector3d origin_R00 = backend::Utility::R2ypr(mRotCam0);
  double y_diff = origin_R0.x() - origin_R00.x();

  // as the optimization may change the pose of all the frames
  // if the first frames pose changed, this will lead to the system random walk behaviour
  // to solve this, we calculate the difference between the frist frame pose before and
  // after the optimization, then propragate it to all other window frames
  //      the system has one rotation DOF and three poisition DOF
  Matrix3d rot_diff = backend::Utility::ypr2R(Vector3d(y_diff, 0, 0));
  if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0) {
    rot_diff = Rs[0] * mRotCam0.transpose();
  }
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

  // maintain marg
  TicToc t_whole_marginalization;
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
