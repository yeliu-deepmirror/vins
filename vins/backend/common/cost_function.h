#pragma once

#include "vins/backend/common/integration_base.h"
#include "vins/backend/common/utility.h"

namespace vins {
namespace backend {

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

void ComputeImuJacobian(const Eigen::Vector3d& t_w_i1, const Eigen::Quaterniond& r_w_i1,
                        const Eigen::Vector3d& v1, const Eigen::Vector3d& ba1,
                        const Eigen::Vector3d& bg1, const Eigen::Vector3d& t_w_i2,
                        const Eigen::Quaterniond& r_w_i2, const Eigen::Vector3d& v2,
                        const Eigen::Vector3d& ba2, const Eigen::Vector3d& bg2,
                        IntegrationBase* pre_integration, Eigen::MatrixXd* jaco_p1,
                        Eigen::MatrixXd* jaco_sb1, Eigen::MatrixXd* jaco_p2,
                        Eigen::MatrixXd* jaco_sb2, Eigen::VectorXd* residual);

}  // namespace backend
}  // namespace vins
