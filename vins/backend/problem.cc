
#include "vins/backend/problem.h"

#include "vins/backend/edge/edge_reprojection.h"

using namespace std;

namespace vins {
namespace backend {
namespace {
bool IsPoseVertex(const std::shared_ptr<vins::backend::Vertex>& v) {
  VertexEdgeTypes type = v->TypeId();
  return type == V_CAMERA_POSE || type == V_IMU_SPEED_BIAS;
}

bool IsLandmarkVertex(const std::shared_ptr<vins::backend::Vertex>& v) {
  VertexEdgeTypes type = v->TypeId();
  return type == V_POINT_XYZ || type == V_INVERSE_DEPTH;
}
}  // namespace

Problem::Problem(ProblemType problem_type) : problem_type_(problem_type) {}

Problem::~Problem() { global_vertex_id = 0; }

bool Problem::AddVertex(std::shared_ptr<Vertex> vertex) {
  if (vertices_.find(vertex->Id()) != vertices_.end()) {
    return false;
  }
  vertices_.insert(std::make_pair(vertex->Id(), vertex));

  if (problem_type_ == ProblemType::SLAM_PROBLEM) {
    if (IsPoseVertex(vertex)) {
      ResizePoseHessiansWhenAddingPose(vertex);
    }
  }
  return true;
}

void Problem::ResizePoseHessiansWhenAddingPose(shared_ptr<Vertex> v) {
  int size = H_prior_.rows() + v->LocalDimension();
  H_prior_.conservativeResize(size, size);
  b_prior_.conservativeResize(size);

  b_prior_.tail(v->LocalDimension()).setZero();
  H_prior_.rightCols(v->LocalDimension()).setZero();
  H_prior_.bottomRows(v->LocalDimension()).setZero();
}
void Problem::ExtendHessiansPriorSize(int dim) {
  int size = H_prior_.rows() + dim;
  H_prior_.conservativeResize(size, size);
  b_prior_.conservativeResize(size);

  b_prior_.tail(dim).setZero();
  H_prior_.rightCols(dim).setZero();
  H_prior_.bottomRows(dim).setZero();
}

bool Problem::AddEdge(shared_ptr<Edge> edge) {
  if (edges_.find(edge->Id()) == edges_.end()) {
    edges_.insert(pair<ulong, std::shared_ptr<Edge>>(edge->Id(), edge));
  } else {
    return false;
  }

  for (auto& vertex : edge->Vertices()) {
    vertexToEdge_.insert(pair<ulong, shared_ptr<Edge>>(vertex->Id(), edge));
  }
  return true;
}

vector<shared_ptr<Edge>> Problem::GetConnectedEdges(std::shared_ptr<Vertex> vertex) {
  vector<shared_ptr<Edge>> edges;
  auto range = vertexToEdge_.equal_range(vertex->Id());
  for (auto iter = range.first; iter != range.second; ++iter) {
    // the edge needs to be existm, and not be removed
    if (edges_.find(iter->second->Id()) == edges_.end()) continue;

    edges.emplace_back(iter->second);
  }
  return edges;
}

bool Problem::RemoveVertex(std::shared_ptr<Vertex> vertex) {
  // check if the vertex is in map_vertices_
  if (vertices_.find(vertex->Id()) == vertices_.end()) {
    return false;
  }

  // delete the edges connected with the vertex
  vector<shared_ptr<Edge>> remove_edges = GetConnectedEdges(vertex);

  for (size_t i = 0; i < remove_edges.size(); i++) {
    RemoveEdge(remove_edges[i]);
  }

  if (IsPoseVertex(vertex))
    idx_pose_vertices_.erase(vertex->Id());
  else
    idx_landmark_vertices_.erase(vertex->Id());

  vertex->SetOrderingId(-1);  // used to debug
  vertices_.erase(vertex->Id());
  vertexToEdge_.erase(vertex->Id());

  return true;
}

bool Problem::RemoveEdge(std::shared_ptr<Edge> edge) {
  // check if the edge is in map_edges_
  if (edges_.find(edge->Id()) == edges_.end()) {
    return false;
  }

  edges_.erase(edge->Id());
  return true;
}

bool Problem::Solve(int iterations, bool verbose) {
  if (edges_.size() == 0 || vertices_.size() == 0) {
    return false;
  }
  TicToc t_solve;
  // compute the dimensions of the system, for later build hessian
  SetOrdering();
  // make the hessian from all the edges
  MakeHessian();
  // LM initizalization
  ComputeLambdaInitLM();

  // LM main solve iteratin
  double last_chi = std::numeric_limits<double>::max();
  if (verbose) {
    std::printf(" | %10s | %13s | %13s | %13s |\n", "Iteration", "Residual", "Norm Dx", "Lambda");
  }
  for (int iter = 0; iter < iterations; iter++) {
    for (int lm_try = 0; lm_try < 8; lm_try++) {
      if (!SolveLinearSystem()) {
        ComputeLambdaInitLM();
        continue;
      }
      UpdateStates();
      if (IsGoodStepInLM()) {
        break;
      }
      // it was a bad update roll back to previous state
      RollbackStates();
    }

    MakeHessian();

    if (verbose) {
      std::printf(" | %10d | %13.6f | %13.6f | %13.6f |\n", iter, currentChi_, dCurrenNormDx_,
                  currentLambda_);
    }

    // if the current residual improvement too small or maybe the linear system solved failed
    if (last_chi - currentChi_ < 1e-6) {
      if (verbose) {
        std::cout << " [LM STOP] the residual reduce too less (< 1e-6)" << std::endl;
      }
      last_chi = currentChi_;
      break;
    }
    last_chi = currentChi_;
  }

  if (verbose) {
    double time_solve = t_solve.toc();
    std::cout << " [LM FINISH]  problem solve cost: " << time_solve << " ms" << std::endl;
    std::cout << " [LM FINISH]  makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;
  }

  t_hessian_cost_ = 0.;
  return true;
}

void Problem::AddOrderingSLAM(std::shared_ptr<vins::backend::Vertex> vertex) {
  if (IsPoseVertex(vertex)) {
    vertex->SetOrderingId(ordering_poses_);
    idx_pose_vertices_.insert(std::make_pair(vertex->Id(), vertex));
    ordering_poses_ += vertex->LocalDimension();
  } else if (IsLandmarkVertex(vertex)) {
    vertex->SetOrderingId(ordering_landmarks_);
    ordering_landmarks_ += vertex->LocalDimension();
    idx_landmark_vertices_.insert(std::make_pair(vertex->Id(), vertex));
  }
}

void Problem::SetOrdering() {
  // reset the ordering parameters
  ordering_poses_ = 0;
  ordering_generic_ = 0;
  ordering_landmarks_ = 0;

  for (auto& vertex : vertices_) {
    ordering_generic_ += vertex.second->LocalDimension();  // the dimension of oprimization
  }

  if (problem_type_ != ProblemType::SLAM_PROBLEM) {
    return;
  }

  for (auto& vertex : vertices_) {
    AddOrderingSLAM(vertex.second);
  }
  ulong all_pose_dimension = ordering_poses_;
  for (auto landmarkVertex : idx_landmark_vertices_) {
    landmarkVertex.second->SetOrderingId(landmarkVertex.second->OrderingId() + all_pose_dimension);
  }
}

void Problem::MakeHessian() {
  TicToc t_h;
  // build the large Hessian matrix
  ulong size = ordering_generic_;
  MatXX H(MatXX::Zero(size, size));
  VecX b(VecX::Zero(size));

  for (auto& edge : edges_) {
    edge.second->ComputeResidual();
    edge.second->ComputeJacobians();

    auto jacobians = edge.second->Jacobians();
    auto vertices = edge.second->Vertices();
    assert(jacobians.size() == vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
      auto v_i = vertices[i];
      // if the vertex is fixed -> should not change its value -> set its jacobian to be zeros
      if (v_i->IsFixed()) continue;

      auto jacobian_i = jacobians[i];
      ulong index_i = v_i->OrderingId();
      ulong dim_i = v_i->LocalDimension();

      // set the robust cost function
      double drho;
      MatXX robustInfo;
      edge.second->RobustInfo(&drho, &robustInfo);

      MatXX JtW = jacobian_i.transpose() * robustInfo;
      for (size_t j = i; j < vertices.size(); ++j) {
        auto v_j = vertices[j];

        if (v_j->IsFixed()) continue;

        auto jacobian_j = jacobians[j];
        ulong index_j = v_j->OrderingId();
        ulong dim_j = v_j->LocalDimension();

        assert(v_j->OrderingId() != -1);
        MatXX hessian = JtW * jacobian_j;

        // acculumate all together
        H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
        if (j != i) {
          H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
        }
      }
      b.segment(index_i, dim_i).noalias() -=
          drho * jacobian_i.transpose() * edge.second->Information() * edge.second->Residual();
    }
  }
  Hessian_ = H;
  b_ = b;
  t_hessian_cost_ += t_h.toc();

  if (H_prior_.rows() > 0) {
    MatXX H_prior_tmp = H_prior_;
    VecX b_prior_tmp = b_prior_;

    /// for all the pose vertices, set its prior dimension to be zero, and fix its external
    /// parameters landmark has no prior
    for (auto vertex : vertices_) {
      if (IsPoseVertex(vertex.second) && vertex.second->IsFixed()) {
        int idx = vertex.second->OrderingId();
        int dim = vertex.second->LocalDimension();
        H_prior_tmp.block(idx, 0, dim, H_prior_tmp.cols()).setZero();
        H_prior_tmp.block(0, idx, H_prior_tmp.rows(), dim).setZero();
        b_prior_tmp.segment(idx, dim).setZero();
      }
    }
    Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
    b_.head(ordering_poses_) += b_prior_tmp;
  }

  delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;
}

bool Problem::SolveLinearSystem() {
  if (problem_type_ == ProblemType::GENERIC_PROBLEM) {
    // PCG solver
    MatXX H = Hessian_;
    for (int i = 0; i < Hessian_.cols(); ++i) {
      H(i, i) += currentLambda_;
    }
    // delta_x_ = PCGSolver(H, b_, H.rows() * 2);
    delta_x_ = H.ldlt().solve(b_);

  } else {
    //        TicToc t_Hmminv;
    // step1: schur marginalization --> Hpp, bpp
    int reserve_size = ordering_poses_;
    int marg_size = ordering_landmarks_;
    MatXX Hmm = Hessian_.block(reserve_size, reserve_size, marg_size, marg_size);
    MatXX Hpm = Hessian_.block(0, reserve_size, reserve_size, marg_size);
    MatXX Hmp = Hessian_.block(reserve_size, 0, marg_size, reserve_size);
    VecX bpp = b_.segment(0, reserve_size);
    VecX bmm = b_.segment(reserve_size, marg_size);

    /*
    // Hmm is diagonal matrix
    MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
    for (auto landmarkVertex : idx_landmark_vertices_) {
        int idx = landmarkVertex.second->OrderingId() - reserve_size;
        int size = landmarkVertex.second->LocalDimension();
        Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
    }
    */

    // BASTIAN_M : use sparse matrix to accelerate and save memory
    // tutorial : https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
    // as for vins system, we use inverse depth, each vertex will only have one value
    // we will reserve the space for memory savage
    Eigen::SparseMatrix<double> Hmm_inv(marg_size, marg_size);
    Hmm_inv.reserve(Eigen::VectorXi::Constant(marg_size, 1));
    for (auto landmarkVertex : idx_landmark_vertices_) {
      int idx = landmarkVertex.second->OrderingId() - reserve_size;
      // BASTIAN_MARK: this is only for vins system
      // int size = landmarkVertex.second->LocalDimension();
      // assert(size == 1);
      Hmm_inv.insert(idx, idx) = 1 / Hmm(idx, idx);
    }
    Hmm_inv.makeCompressed();

    MatXX tempH = Hpm * Hmm_inv;
    H_pp_schur_ = Hessian_.block(0, 0, ordering_poses_, ordering_poses_) - tempH * Hmp;
    b_pp_schur_ = bpp - tempH * bmm;

    // step2: solve Hpp * delta_x = bpp
    VecX delta_x_pp(VecX::Zero(reserve_size));

    for (ulong i = 0; i < ordering_poses_; ++i) {
      H_pp_schur_(i, i) += currentLambda_;  // LM Method
    }

    // Utility::SaveMatrixToFile(H_pp_schur_, b_pp_schur_);

    // TicToc t_linearsolver;
    delta_x_pp = H_pp_schur_.ldlt().solve(b_pp_schur_);  //  SVec.asDiagonal() * svd.matrixV() * Ub;
    // std::cout << " Linear Solver Time Cost: " << t_linearsolver.toc() << std::endl;

    // TicToc t_linearsolver_pcg;
    // delta_x_pp = PCGSolver(H_pp_schur_, b_pp_schur_, H_pp_schur_.rows()/2, 1e-7);
    // std::cout << " PCG Linear Solver Time Cost: " << t_linearsolver_pcg.toc() << std::endl;

    delta_x_.head(reserve_size) = delta_x_pp;

    // step3: solve Hmm * delta_x = bmm - Hmp * delta_x_pp;
    VecX delta_x_ll(marg_size);
    delta_x_ll = Hmm_inv * (bmm - Hmp * delta_x_pp);
    delta_x_.tail(marg_size) = delta_x_ll;
  }
  dCurrenNormDx_ = delta_x_.norm();
  // solve failed, as the delta x is much too large
  return dCurrenNormDx_ < 1e3;
}

void Problem::UpdateStates() {
  // update vertex
  for (auto vertex : vertices_) {
    vertex.second->BackUpParameters();  // save the last estimation

    ulong idx = vertex.second->OrderingId();
    ulong dim = vertex.second->LocalDimension();
    VecX delta = delta_x_.segment(idx, dim);
    vertex.second->Plus(delta);
  }

  // update prior
  if (err_prior_.rows() > 0) {
    // BACK UP b_prior_
    b_prior_backup_ = b_prior_;
    err_prior_backup_ = err_prior_;

    /// update with first order Taylor, b' = b + \frac{\delta b}{\delta x} * \delta x
    /// \delta x = Computes the linearized deviation from the references (linearization points)
    b_prior_ -= H_prior_ * delta_x_.head(ordering_poses_);  // update the error_prior
    err_prior_ = -Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 15);
  }
}

void Problem::RollbackStates() {
  // update vertex
  for (auto vertex : vertices_) {
    vertex.second->RollBackParameters();
  }

  // Roll back prior_
  if (err_prior_.rows() > 0) {
    b_prior_ = b_prior_backup_;
    err_prior_ = err_prior_backup_;
  }
}

/// LM
void Problem::ComputeLambdaInitLM() {
  ni_ = 2.;
  currentLambda_ = -1.;
  currentChi_ = 0.0;

  for (auto edge : edges_) {
    currentChi_ += edge.second->RobustCost();
  }
  if (err_prior_.rows() > 0) currentChi_ += err_prior_.norm();
  currentChi_ *= 0.5;

  // not use this stop threshold
  // stopThresholdLM_ = 1e-10 * currentChi_;

  double maxDiagonal = 0;
  ulong size = Hessian_.cols();
  // assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
  for (ulong i = 0; i < size; ++i) {
    maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal);
  }

  maxDiagonal = std::min(5e10, maxDiagonal);
  double tau = 1e-5;  // 1e-5
  currentLambda_ = tau * maxDiagonal;
}

// not used in slam system
void Problem::AddLambdatoHessianLM() {
  ulong size = Hessian_.cols();
  assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
  for (ulong i = 0; i < size; ++i) {
    Hessian_(i, i) += currentLambda_;
  }
}

// not used in slam system
void Problem::RemoveLambdaHessianLM() {
  ulong size = Hessian_.cols();
  assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
  for (ulong i = 0; i < size; ++i) {
    Hessian_(i, i) -= currentLambda_;
  }
}

bool Problem::IsGoodStepInLM() {
  double scale = 0;
  scale = 0.5 * delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
  scale += 1e-6;  // make sure it's non-zero :)

  // recompute residuals after update state
  double tempChi = 0.0;
  for (auto edge : edges_) {
    edge.second->ComputeResidual();
    tempChi += edge.second->RobustCost();
  }
  if (err_prior_.size() > 0) tempChi += err_prior_.norm();
  tempChi *= 0.5;  // 1/2 * err^2

  double rho = (currentChi_ - tempChi) / scale;
  if (rho > 0 && isfinite(tempChi))  // last step was good
  {
    double alpha = 1. - pow((2 * rho - 1), 3);
    alpha = std::min(alpha, 2. / 3.);
    double scaleFactor = (std::max)(1. / 3., alpha);
    currentLambda_ *= scaleFactor;
    ni_ = 2;
    currentChi_ = tempChi;
    return true;
  } else {
    currentLambda_ *= ni_;
    ni_ *= 2;
    return false;
  }
}

/** @brief conjugate gradient with perconditioning
 *  from my matlab implementation
 *  https://gitee.com/gggliuye/cg_pcg/blob/master/mine_pcg.m
 *  input should be a square matrix
 *
 *  TODO: not assured to converge
 */
VecX Problem::PCGSolver(const MatXX& Hpp, const VecX& bpp, int maxIter, double stop_threshold) {
  int n = Hpp.cols();
  VecX x(VecX::Zero(n));
  VecX r(bpp);
  double stop_rho = stop_threshold * sqrt(r.dot(r));
  stop_rho = stop_rho * stop_rho;

  // build the preconditonor, use sparse matrix to accelerate.
  Eigen::SparseMatrix<double> M_inv_sparse(n, n);
  for (int i = 0; i < n; i++) {
    M_inv_sparse.insert(i, i) = 1 / Hpp(i, i);
  }

  VecX z = M_inv_sparse * r;
  double rho = r.dot(z);
  double last_rho = rho;

  VecX p(VecX::Zero(n));
  int k;
  for (k = 0; k < maxIter; k++) {
    if (rho < stop_rho) {
      break;
    }
    if (k == 0) {
      p = M_inv_sparse * r;
    } else {
      p = z + (rho / last_rho) * p;
    }
    VecX w = Hpp * p;
    double alpha = rho / (p.dot(w));
    x += alpha * p;
    r -= alpha * w;
    z = M_inv_sparse * r;
    last_rho = rho;
    rho = r.dot(z);
  }
  return x;
}

/*
 * marginalize all the edges connected with the frame (imu factors and the projection factors)
 *
 * If we want to keep a landmark which is connected with the frame, we should first delete this
 * edge, otherwise the Hessian will become denser, which will slow down the process.
 */
bool Problem::Marginalize(const std::vector<std::shared_ptr<Vertex>> margVertexs, int pose_dim) {
  SetOrdering();
  /// find the edges to marginalize,
  // margVertexs[0] is frame, its edge contained pre-intergration
  std::vector<shared_ptr<Edge>> marg_edges = GetConnectedEdges(margVertexs[0]);

  // when find the marginalized landmark, find its 3D position in world and save it
  marginalized_pts.clear();
  current_pts.clear();

  std::unordered_map<int, shared_ptr<Vertex>> margLandmark;

  // build the Hessian matrix while keep the order of pose
  // the order of landmarks may change
  int marg_landmark_size = 0;

  for (size_t i = 0; i < marg_edges.size(); ++i) {
    if (marg_edges[i]->TypeInfo() == string("EdgeReprojection")) {
      // only the first vertices of edge reprojection is landmark vertex
      // and we are sure that this will contain all the vertices we want
      auto iter = marg_edges[i]->Vertices()[0];
      if (margLandmark.find(iter->Id()) == margLandmark.end()) {
        iter->SetOrderingId(pose_dim + marg_landmark_size);
        margLandmark.insert(make_pair(iter->Id(), iter));
        // marg_landmark_size += iter->LocalDimension();
        // LocalDimension() is one for landmark vertices (inverse depth only)
        marg_landmark_size += 1;

        if (ifRecordMap) {
          // the vertex[1] of the edge i should be the host point from which we can extract its
          // world position here we saved all the points connected to the frame however we should
          // only save the points only owned by the frame
          std::vector<std::shared_ptr<Edge>> remove_edges = GetConnectedEdges(iter);
          std::shared_ptr<EdgeReprojection> edge_factor =
              std::static_pointer_cast<EdgeReprojection>(marg_edges[i]);
          Vec3 pts_w = edge_factor->GetPointInWorld();
          if (remove_edges.size() == 1) {
            marginalized_pts.push_back(pts_w);
          } else {
            current_pts.push_back(pts_w);
          }
        }
      }
    }
    /*
            // old version, loop all the vertices to find landmark vertices
            auto vertices = marg_edges[i]->Vertices();
            for (auto iter : vertices) {
                // if the vertex is a landmark and is not redundant in the margLandmark set
                if (IsLandmarkVertex(iter) && margLandmark.find(iter->Id()) == margLandmark.end()) {
                    iter->SetOrderingId(pose_dim + marg_landmark_size);
                    margLandmark.insert(make_pair(iter->Id(), iter));
                    marg_landmark_size += iter->LocalDimension();
                }
            }
    */
  }

  // the size of variables to be marginalized // pose_dim = 171
  int cols = pose_dim + marg_landmark_size;

  /// build the hessian matrix:  H = H_marg + H_pp_prior
  // H_marg has variables : { external_parameters,  poses , landmark_view_by_marginalized_frame )
  MatXX H_marg(MatXX::Zero(cols, cols));
  VecX b_marg(VecX::Zero(cols));
  int ii = 0;
  for (auto edge : marg_edges) {
    edge->ComputeResidual();
    edge->ComputeJacobians();
    auto jacobians = edge->Jacobians();
    auto vertices = edge->Vertices();
    ii++;

    assert(jacobians.size() == vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
      auto v_i = vertices[i];
      auto jacobian_i = jacobians[i];
      ulong index_i = v_i->OrderingId();
      ulong dim_i = v_i->LocalDimension();

      double drho;
      MatXX robustInfo;
      edge->RobustInfo(&drho, &robustInfo);

      for (size_t j = i; j < vertices.size(); ++j) {
        auto v_j = vertices[j];
        auto jacobian_j = jacobians[j];
        ulong index_j = v_j->OrderingId();
        ulong dim_j = v_j->LocalDimension();

        MatXX hessian = jacobian_i.transpose() * robustInfo * jacobian_j;

        assert(hessian.rows() == v_i->LocalDimension() && hessian.cols() == v_j->LocalDimension());
        // add all the hessians
        H_marg.block(index_i, index_j, dim_i, dim_j) += hessian;
        if (j != i) {
          // hessian is symmetry
          H_marg.block(index_j, index_i, dim_j, dim_i) += hessian.transpose();
        }
      }
      b_marg.segment(index_i, dim_i) -=
          drho * jacobian_i.transpose() * edge->Information() * edge->Residual();
    }
  }

  /// marg landmark
  int reserve_size = pose_dim;
  if (marg_landmark_size > 0) {
    int marg_size = marg_landmark_size;
    MatXX Hmm = H_marg.block(reserve_size, reserve_size, marg_size, marg_size);
    MatXX Hpm = H_marg.block(0, reserve_size, reserve_size, marg_size);
    MatXX Hmp = H_marg.block(reserve_size, 0, marg_size, reserve_size);
    VecX bpp = b_marg.segment(0, reserve_size);
    VecX bmm = b_marg.segment(reserve_size, marg_size);

    /*
    // Hmm is digonal matrix, its inverse can be calculated by its blocks' inverses.
    MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
    // TODO:: use openMP to speed up
    for (auto iter: margLandmark) {
        int idx = iter.second->OrderingId() - reserve_size;
        int size = iter.second->LocalDimension();
        Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
    }
    */

    // BASTIAN_M : use sparse matrix to accelerate and save memory
    // tutorial : https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
    // as for vins system, we use inverse depth, each vertex will only have one value
    // we will reserve the space for memory savage
    Eigen::SparseMatrix<double> Hmm_inv(marg_size, marg_size);
    Hmm_inv.reserve(Eigen::VectorXi::Constant(marg_size, 1));
    for (auto landmarkVertex : idx_landmark_vertices_) {
      int idx = landmarkVertex.second->OrderingId() - reserve_size;
      // BASTIAN_MARK: this is only for vins system
      // int size = landmarkVertex.second->LocalDimension();
      // assert(size == 1);
      Hmm_inv.insert(idx, idx) = 1 / Hmm(idx, idx);
    }
    Hmm_inv.makeCompressed();

    MatXX tempH = Hpm * Hmm_inv;
    MatXX Hpp = H_marg.block(0, 0, reserve_size, reserve_size) - tempH * Hmp;
    bpp = bpp - tempH * bmm;
    H_marg = Hpp;
    b_marg = bpp;
  }

  VecX b_prior_before = b_prior_;
  if (H_prior_.rows() > 0) {
    H_marg += H_prior_;
    b_marg += b_prior_;
  }

  /// marg frame and speedbias
  int marg_dim = 0;

  // move the index from the tail of the vector
  for (int k = margVertexs.size() - 1; k >= 0; --k) {
    int idx = margVertexs[k]->OrderingId();
    int dim = margVertexs[k]->LocalDimension();
    marg_dim += dim;
    // move the marginalized frame pose to the Hmm bottown right
    // 1. move row i the lowest part
    Eigen::MatrixXd temp_rows = H_marg.block(idx, 0, dim, reserve_size);
    Eigen::MatrixXd temp_botRows =
        H_marg.block(idx + dim, 0, reserve_size - idx - dim, reserve_size);
    H_marg.block(idx, 0, reserve_size - idx - dim, reserve_size) = temp_botRows;
    H_marg.block(reserve_size - dim, 0, dim, reserve_size) = temp_rows;

    // put col i to the rightest part
    Eigen::MatrixXd temp_cols = H_marg.block(0, idx, reserve_size, dim);
    Eigen::MatrixXd temp_rightCols =
        H_marg.block(0, idx + dim, reserve_size, reserve_size - idx - dim);
    H_marg.block(0, idx, reserve_size, reserve_size - idx - dim) = temp_rightCols;
    H_marg.block(0, reserve_size - dim, reserve_size, dim) = temp_cols;

    Eigen::VectorXd temp_b = b_marg.segment(idx, dim);
    Eigen::VectorXd temp_btail = b_marg.segment(idx + dim, reserve_size - idx - dim);
    b_marg.segment(idx, reserve_size - idx - dim) = temp_btail;
    b_marg.segment(reserve_size - dim, dim) = temp_b;
  }

  double eps = 1e-8;
  int m2 = marg_dim;
  int n2 = reserve_size - marg_dim;  // marg pose
  Eigen::MatrixXd Amm =
      0.5 * (H_marg.block(n2, n2, m2, m2) + H_marg.block(n2, n2, m2, m2).transpose());

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
  Eigen::MatrixXd Amm_inv =
      saes.eigenvectors() *
      Eigen::VectorXd(
          (saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0))
          .asDiagonal() *
      saes.eigenvectors().transpose();

  Eigen::VectorXd bmm2 = b_marg.segment(n2, m2);
  Eigen::MatrixXd Arm = H_marg.block(0, n2, n2, m2);
  Eigen::MatrixXd Amr = H_marg.block(n2, 0, m2, n2);
  Eigen::MatrixXd Arr = H_marg.block(0, 0, n2, n2);
  Eigen::VectorXd brr = b_marg.segment(0, n2);
  Eigen::MatrixXd tempB = Arm * Amm_inv;
  // the rest pose become the new prior matrix
  // upper we move the rest pose to the upper left, and new pose will be added at lower right.
  // as a result the order of variables of the prior matrix is corresponding to the up coming
  // Hessian.
  H_prior_ = Arr - tempB * Amr;
  b_prior_ = brr - tempB * bmm2;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(H_prior_);
  Eigen::VectorXd S =
      Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
  Eigen::VectorXd S_inv = Eigen::VectorXd(
      (saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

  Eigen::VectorXd S_sqrt = S.cwiseSqrt();
  Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
  Jt_prior_inv_ = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
  err_prior_ = -Jt_prior_inv_ * b_prior_;

  MatXX J = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
  H_prior_ = J.transpose() * J;
  MatXX tmp_h = MatXX((H_prior_.array().abs() > 1e-9).select(H_prior_.array(), 0));
  H_prior_ = tmp_h;

  // remove vertex and remove edge
  for (size_t k = 0; k < margVertexs.size(); ++k) {
    RemoveVertex(margVertexs[k]);
  }

  for (auto landmarkVertex : margLandmark) {
    RemoveVertex(landmarkVertex.second);
  }

  return true;
}

}  // namespace backend
}  // namespace vins
