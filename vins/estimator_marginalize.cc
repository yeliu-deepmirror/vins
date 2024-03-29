
#include "vins/estimator.h"

namespace vins {

void Estimator::MargOldFrame() {
  std::shared_ptr<backend::LossFunction> lossfunction = std::make_shared<backend::CauchyLoss>(1.0);

  // step1. build problem
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
    problem.AddVertex(vertexExt);
    pose_dim += vertexExt->LocalDimension();
  }

  // add the camera pose vertexs
  for (int i = 0; i < feature::WINDOW_SIZE + 1; i++) {
    shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
    Eigen::VectorXd pose(7);
    Quaterniond q_init(Rs[i]);
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

  // add IMU preintegration edges
  {
    std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
    edge_vertex.push_back(vertexCams_vec[0]);
    edge_vertex.push_back(vertexVB_vec[0]);
    edge_vertex.push_back(vertexCams_vec[1]);
    edge_vertex.push_back(vertexVB_vec[1]);
    if (pre_integrations[1]->sum_dt < 10.0) {
      std::shared_ptr<backend::EdgeImu> imuEdge =
          std::make_shared<backend::EdgeImu>(pre_integrations[1], edge_vertex);
      problem.AddEdge(imuEdge);
    }
  }

  // Visual Factor
  {
    int feature_index = -1;
    // for all the viewed features
    for (auto& it_per_id : f_manager.feature) {
      it_per_id.used_num = it_per_id.feature_per_frame.size();
      if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < feature::WINDOW_SIZE - 2)) continue;

      ++feature_index;

      int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
      if (imu_i != 0) continue;

      const Vector3d& pts_i = it_per_id.feature_per_frame[0].point;

      shared_ptr<backend::VertexInverseDepth> verterxPoint(new backend::VertexInverseDepth());
      VecX inv_d(1);
      inv_d << vInverseDepth[feature_index];
      verterxPoint->SetParameters(inv_d);
      problem.AddVertex(verterxPoint);

      if (save_pts_) {
        // add point to map
        Eigen::Vector3d pt_camera_1 = pts_i / vInverseDepth[feature_index];
        all_map_points_[it_per_id.feature_id] = Rs[imu_i] * (rigid_ic_ * pt_camera_1) + Ps[imu_i];
      }

      if (it_per_id.inv_depth_gt_.has_value()) {
        // if has depth ground truth add factor for it
        std::vector<std::shared_ptr<backend::Vertex>> edge_vertex{verterxPoint};
        std::shared_ptr<backend::EdgeInvDepthPrior> edge =
            std::make_shared<backend::EdgeInvDepthPrior>(it_per_id.inv_depth_gt_.value(),
                                                         edge_vertex);
        edge->SetInformation(depth_information_);
        problem.AddEdge(edge);
      }

      // for all its observations -> each has a reprojection error w.r.t the first observation
      for (auto& it_per_frame : it_per_id.feature_per_frame) {
        imu_j++;
        if (imu_i == imu_j) continue;

        const Vector3d& pts_j = it_per_frame.point;

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

  // prior observations (the vertex and edges that are marginalized)
  {
    // If it is already exist
    if (Hprior_.rows() > 0) {
      problem.SetHessianPrior(Hprior_);  // set the prior matrix
      problem.SetbPrior(bprior_);
      problem.SetErrPrior(errprior_);
      problem.SetJtPrior(Jprior_inv_);
      problem.ExtendHessiansPriorSize(15);  // extend the prior matrix for new marginalization
    } else {
      Hprior_ = MatXX(pose_dim, pose_dim);
      Hprior_.setZero();
      bprior_ = VecX(pose_dim);
      bprior_.setZero();
      problem.SetHessianPrior(Hprior_);  // set the initial prior matrix
      problem.SetbPrior(bprior_);
    }
  }

  // build the marginalization elements
  std::vector<std::shared_ptr<backend::Vertex>> marg_vertex;
  marg_vertex.push_back(vertexCams_vec[0]);
  marg_vertex.push_back(vertexVB_vec[0]);

  // marginalize the old camera
  problem.Marginalize(marg_vertex, pose_dim);

  Hprior_ = problem.GetHessianPrior();
  bprior_ = problem.GetbPrior();
  errprior_ = problem.GetErrPrior();
  Jprior_inv_ = problem.GetJtPrior();
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
  CHECK(Hprior_.rows() > 0);
  ExtendedPrior(15);
  backend::MarginalizeFrameInternal(6 + (feature::WINDOW_SIZE - 1) * 15, 15, &Hprior_, &bprior_,
                                    &errprior_, &Jprior_inv_);
  // if (!estimate_extrinsics_) {
  //   Hprior_.block(0, 0, 6, Hprior_.cols()).setZero();
  //   Hprior_.block(0, 0, Hprior_.rows(), 6).setZero();
  //   bprior_.segment(0, 6).setZero();
  // }
}

}  // namespace vins
