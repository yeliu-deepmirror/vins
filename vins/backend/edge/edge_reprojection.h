#pragma once

#include "vins/backend/common/utility.h"
#include "vins/backend/edge/edge.h"

namespace vins {
namespace backend {

class EdgeReprojection : public Edge {
 public:
  EdgeReprojection(const Eigen::Vector3d& pts_i, const Eigen::Vector3d& pts_j,
                   const std::vector<std::shared_ptr<Vertex>>& vertices);

  std::string TypeInfo() const override { return "EdgeReprojection"; }
  void ComputeResidual() override;
  void ComputeJacobians() override;

  Eigen::Vector3d GetPointInWorld();

 private:
  Eigen::Vector3d pts_i_;
  Eigen::Vector3d pts_j_;
};

Eigen::Vector2d ComputeVisualResidual(const Eigen::Vector3d& t_i_c, const Eigen::Matrix3d& r_i_c,
                                      const Eigen::Vector3d& t_w_i1, const Eigen::Matrix3d& r_w_i1,
                                      const Eigen::Vector3d& t_w_i2, const Eigen::Matrix3d& r_w_i2,
                                      double inv_dep_1, const Eigen::Vector3d& pt1,
                                      const Eigen::Vector3d& pt2);

void ComputeVisualJacobian(const Eigen::Vector3d& t_i_c, const Eigen::Matrix3d& r_i_c,
                           const Eigen::Vector3d& t_w_i1, const Eigen::Matrix3d& r_w_i1,
                           const Eigen::Vector3d& t_w_i2, const Eigen::Matrix3d& r_w_i2,
                           double inv_dep_1, const Eigen::Vector3d& pt1, const Eigen::Vector3d& pt2,
                           Eigen::MatrixXd* jaco_1, Eigen::MatrixXd* jaco_2,
                           Eigen::MatrixXd* jaco_ex, Eigen::MatrixXd* jaco_pt,
                           Eigen::VectorXd* residual = nullptr);

}  // namespace backend
}  // namespace vins
