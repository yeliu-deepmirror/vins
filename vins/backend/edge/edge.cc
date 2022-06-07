#include "vins/backend/edge/edge.h"
#include "vins/backend/vertex/vertex.h"

using namespace std;

namespace vins {
namespace backend {

unsigned long global_edge_id = 0;

Edge::Edge(int residual_dimension, int num_vertices,
           const std::vector<std::shared_ptr<Vertex>>& vertices,
           const std::vector<VertexEdgeTypes>& vertices_types)
    : vertices_types_(vertices_types), vertices_(vertices), lossfunction_(nullptr) {
  CHECK_EQ(vertices_.size(), vertices_types_.size());
  residual_.resize(residual_dimension, 1);
  jacobians_.resize(num_vertices);
  id_ = global_edge_id++;

  information_ = Eigen::MatrixXd(residual_dimension, residual_dimension);
  information_.setIdentity();
  sqrt_information_ = information_;
}

Edge::~Edge() {}

void Edge::Check() {
  CHECK_EQ(vertices_.size(), vertices_types_.size());
  for (size_t i = 0; i < vertices_.size(); i++) {
    CHECK_EQ(vertices_[i]->TypeId(), vertices_types_[i]) << i;
  }
}

double Edge::Cost() const { return residual_.transpose() * information_ * residual_; }

double Edge::RobustCost() const {
  if (lossfunction_) {
    return lossfunction_->Compute(this->Cost())(0);
  }
  return this->Cost();
}

void Edge::RobustInfo(double* drho, Eigen::MatrixXd* info) const {
  if (lossfunction_) {
    RobustInformation(*lossfunction_.get(), information_, sqrt_information_, residual_, drho, info);
  } else {
    *drho = 1.0;
    *info = information_;
  }
}

void Edge::SetInformation(const Eigen::MatrixXd& information) {
  CHECK_EQ(information.cols(), information.rows());
  CHECK_EQ(information.cols(), residual_.size());

  information_ = information;
  sqrt_information_ = Eigen::LLT<Eigen::MatrixXd>(information_).matrixL().transpose();
}

}  // namespace backend
}  // namespace vins
