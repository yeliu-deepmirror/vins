#include <iostream>
#include <random>

#include "gtest/gtest.h"
#include "vins/backend/problem.h"

using namespace std;

namespace vins::backend {

class CurveFittingVertex : public Vertex {
 public:
  CurveFittingVertex() : Vertex(3) {}
  VertexEdgeTypes TypeId() const { return V_COMMON; }
};

class CurveFittingEdge : public Edge {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CurveFittingEdge(double x, double y, const std::vector<std::shared_ptr<Vertex>>& vertices)
      : Edge(1, 1, vertices, std::vector<VertexEdgeTypes>{V_COMMON}) {
    x_ = x;
    y_ = y;
  }
  void ComputeResidual() override {
    Vec3 abc = vertices_[0]->Parameters();
    residual_(0) = std::exp(abc(0) * x_ * x_ + abc(1) * x_ + abc(2)) - y_;
  }
  void ComputeJacobians() override {
    Vec3 abc = vertices_[0]->Parameters();
    double exp_y = std::exp(abc(0) * x_ * x_ + abc(1) * x_ + abc(2));
    Eigen::Matrix<double, 1, 3> jaco_abc;
    jaco_abc << x_ * x_ * exp_y, x_ * exp_y, 1 * exp_y;
    jacobians_[0] = jaco_abc;
  }
  std::string TypeInfo() const override { return "CurveFittingEdge"; }

 public:
  double x_, y_;
};

TEST(ProblemTest, CurveFitting) {
  Eigen::Vector3d parameters_gt = Eigen::Vector3d::Random();
  int num_pts = 100;

  Problem problem(Problem::ProblemType::GENERIC_PROBLEM);
  std::shared_ptr<CurveFittingVertex> vertex = std::make_shared<CurveFittingVertex>();
  vertex->SetParameters(Eigen::Vector3d::Zero());
  problem.AddVertex(vertex);

  std::vector<std::shared_ptr<Vertex>> edge_vertex;
  edge_vertex.push_back(vertex);

  for (int i = 0; i < num_pts; ++i) {
    double x = i / 100.0;
    double y = std::exp(parameters_gt(0) * x * x + parameters_gt(1) * x + parameters_gt(2));

    std::shared_ptr<CurveFittingEdge> edge = std::make_shared<CurveFittingEdge>(x, y, edge_vertex);
    problem.AddEdge(edge);
  }
  problem.Solve(30);

  Eigen::Vector3d error = vertex->Parameters() - parameters_gt;
  EXPECT_NEAR(error.norm(), 0.0, 1e-5);
}

}  // namespace vins::backend
