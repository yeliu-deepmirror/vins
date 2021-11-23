#pragma once

#include <memory>
#include <string>

#include "vins/backend/common/loss_function.h"
#include "vins/backend/vertex/vertex.h"

namespace vins {
namespace backend {

class Edge {
 public:
  Edge(int residual_dimension, int num_vertices,
       const std::vector<std::shared_ptr<Vertex>>& vertices,
       const std::vector<VertexEdgeTypes>& vertices_types);
  virtual ~Edge();

  // check the vertex type, if needed
  // since it takes loops, we won't do it by default
  void Check();

  unsigned long Id() const { return id_; }

  std::shared_ptr<Vertex> GetVertex(int i) { return vertices_[i]; }
  std::vector<std::shared_ptr<Vertex>> Vertices() const { return vertices_; }
  size_t NumVertices() const { return vertices_.size(); }

  virtual std::string TypeInfo() const = 0;
  virtual void ComputeResidual() = 0;
  virtual void ComputeJacobians() = 0;

  /// compute the squared error, which will be timed by infomation matrix
  double Cost() const;
  double RobustCost() const;

  /// return the raw residual and jacobians
  Eigen::VectorXd Residual() const { return residual_; }
  std::vector<Eigen::MatrixXd> Jacobians() const { return jacobians_; }

  void SetInformation(const Eigen::MatrixXd& information);
  Eigen::MatrixXd Information() const { return information_; }
  Eigen::MatrixXd SqrtInformation() const { return sqrt_information_; }

  void SetLossFunction(std::shared_ptr<LossFunction> ptr) { lossfunction_ = ptr; }
  void RobustInfo(double* drho, Eigen::MatrixXd* info) const;

  int OrderingId() const { return ordering_id_; }
  void SetOrderingId(int id) { ordering_id_ = id; };

 protected:
  unsigned long id_;
  int ordering_id_;

  std::vector<VertexEdgeTypes> vertices_types_;
  std::vector<std::shared_ptr<Vertex>> vertices_;

  Eigen::VectorXd residual_;
  std::vector<Eigen::MatrixXd> jacobians_;

  Eigen::MatrixXd information_;
  Eigen::MatrixXd sqrt_information_;

  std::shared_ptr<LossFunction> lossfunction_;
};

}  // namespace backend
}  // namespace vins
