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
    double cost = this->Cost();
    Eigen::Vector3d rho = lossfunction_->Compute(cost);
    Eigen::VectorXd weight_err = sqrt_information_ * residual_;
    Eigen::MatrixXd robust_info(information_.rows(), information_.cols());
    robust_info.setIdentity();
    robust_info *= rho[1];
    if (rho[1] + 2 * rho[2] * cost > 0.) {
      robust_info += 2 * rho[2] * weight_err * weight_err.transpose();
    }
    *info = robust_info * information_;
    *drho = rho[1];
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
