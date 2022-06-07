#ifndef MYSLAM_BACKEND_PROBLEM_H
#define MYSLAM_BACKEND_PROBLEM_H

#include <map>
#include <unordered_map>

#include "glog/logging.h"

#include "vins/backend/common/tic_toc.h"
#include "vins/backend/edge/edge.h"
#include "vins/backend/vertex/vertex.h"

typedef unsigned long ulong;

namespace vins {
namespace backend {

typedef unsigned long ulong;
typedef std::map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
typedef std::unordered_map<unsigned long, std::shared_ptr<Edge>> HashEdge;
typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>> HashVertexIdToEdge;

class Problem {
 public:
  enum class ProblemType { SLAM_PROBLEM, GENERIC_PROBLEM };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Problem(ProblemType problem_type);
  ~Problem();

  bool AddVertex(std::shared_ptr<Vertex> vertex);

  bool AddEdge(std::shared_ptr<Edge> edge);

  void GetOutlierEdges(std::vector<std::shared_ptr<Edge>>& outlier_edges);

  bool Solve(int iterations = 10, bool verbose = true);

  MatXX GetHessianPrior() { return H_prior_; }
  VecX GetbPrior() { return b_prior_; }
  VecX GetErrPrior() { return err_prior_; }
  MatXX GetJtPrior() { return Jt_prior_inv_; }

  void SetHessianPrior(const MatXX& H) { H_prior_ = H; }
  void SetbPrior(const VecX& b) { b_prior_ = b; }
  void SetErrPrior(const VecX& b) { err_prior_ = b; }
  void SetJtPrior(const MatXX& J) { Jt_prior_inv_ = J; }

  void ExtendHessiansPriorSize(int dim);

  Eigen::MatrixXd H_marg_test;
  Eigen::VectorXd b_marg_test;

 private:
  /// Solve SLAM problem
  bool SolveSLAMProblem(int iterations);

  /// set the order of every index : ordering_index
  void SetOrdering();

  /// set ordering for new vertex in slam problem
  void AddOrderingSLAM(std::shared_ptr<Vertex> v);

  /// build the whole hessian matrix
  void MakeHessian();

  /// schur to solve SBA problem
  void SchurSBA();

  /// solve the linea problem
  bool SolveLinearSystem();

  /// update the states
  void UpdateStates();

  // sometimes the residual become larger after update -> bad update
  // require to roll back the former state
  void RollbackStates();

  /// compute and update the prior part
  void ComputePrior();

  /// after adding a new vertex, the size changed of Hessian matrix
  void ResizePoseHessiansWhenAddingPose(std::shared_ptr<Vertex> v);

  /// get the edges connect to a vertex
  std::vector<std::shared_ptr<Edge>> GetConnectedEdges(std::shared_ptr<Vertex> vertex);

  /// Levenberg

  /// initialize the Lambda parameter
  void ComputeLambdaInitLM();

  /// LM whether the lambda is good and how to update it
  bool IsGoodStepInLM();

  /// PCG iteration solver
  VecX PCGSolver(const MatXX& Hpp, const VecX& bpp, int maxIter, double stop_threshold);

 public:
  // for save marginalized points
  bool ifRecordMap = true;
  std::vector<Vec3> marginalized_pts;
  std::vector<Vec3> current_pts;

 private:
  double currentLambda_;
  double currentChi_;
  double dCurrenNormDx_ = 0;
  double stopThresholdLM_;  // LM threshold to quit
  double ni_;               // the control the change of lambda

  ProblemType problem_type_;

  /// the whole hessian
  MatXX Hessian_;
  VecX b_;
  VecX delta_x_;

  /// prior part of the system
  MatXX H_prior_;
  VecX b_prior_;
  VecX b_prior_backup_;
  VecX err_prior_backup_;

  MatXX Jt_prior_inv_;
  VecX err_prior_;

  /// pose of SBA problem
  MatXX H_pp_schur_;
  VecX b_pp_schur_;

  // landmark(ll) and pose(pp) of hessian
  MatXX H_pp_;
  VecX b_pp_;
  MatXX H_ll_;
  VecX b_ll_;

  /// all vertices
  HashVertex vertices_;

  /// all edges
  HashEdge edges_;

  /// find edge by vertex id
  HashVertexIdToEdge vertexToEdge_;

  /// Ordering related
  ulong ordering_poses_ = 0;
  ulong ordering_landmarks_ = 0;
  ulong ordering_generic_ = 0;
  std::map<unsigned long, std::shared_ptr<Vertex>> idx_pose_vertices_;  // order the pose vertex
  std::map<unsigned long, std::shared_ptr<Vertex>>
      idx_landmark_vertices_;  // order the landmark vertex

  // vertices need to marg. <Ordering_id_, Vertex>
  HashVertex vertices_marg_;

  bool bDebug = false;
  double t_hessian_cost_ = 0.0;
  double t_PCGsovle_cost_ = 0.0;
};

}  // namespace backend
}  // namespace vins

#endif
