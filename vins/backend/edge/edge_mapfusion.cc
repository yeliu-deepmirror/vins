
#include "vins/backend/edge/edge_mapfusion.h"
#include "vins/backend/vertex/vertex_pose.h"
#include "vins/backend/vertex/vertex_scale.h"

#define USE_SO3_JACOBIAN 1

namespace vins {
namespace backend {

Qd deltaQ(const Vec3& theta) {
  Eigen::Quaternion<double> dq;
  Eigen::Matrix<double, 3, 1> half_theta = theta;
  half_theta /= 2.0;
  dq.w() = 1.0;
  dq.x() = half_theta.x();
  dq.y() = half_theta.y();
  dq.z() = half_theta.z();
  return dq;
}

void EdgeMapFusion::ComputeResidual() {
  VecX param_pose = vertices_[0]->Parameters();
  Qd Q_relative(param_pose[6], param_pose[3], param_pose[4], param_pose[5]);
  Vec3 P_relative = param_pose.head<3>();

  double scale = vertices_[1]->Parameters()[0];

  residual_.setZero();

  Vec3 vectorErr = -pts_j_ + scale * (Q_relative * pts_i_ + P_relative);
  residual_ = vectorErr;
}

void EdgeMapFusion::ComputeJacobians() {
  VecX param_pose = vertices_[0]->Parameters();
  Qd Q_relative(param_pose[6], param_pose[3], param_pose[4], param_pose[5]);
  Vec3 P_relative = param_pose.head<3>();

  double scale = vertices_[1]->Parameters()[0];

  Eigen::Matrix<double, 3, 1> jacobian_scale;
  jacobian_scale = Q_relative * pts_i_ + P_relative;

  Eigen::Matrix<double, 3, 6> jacobian_relative_pose;
  jacobian_relative_pose.setZero();
  jacobian_relative_pose.leftCols<3>() = scale * Eigen::Matrix3d::Identity();
  jacobian_relative_pose.rightCols<3>() = -scale * (Q_relative * Sophus::SO3d::hat(pts_i_));

  jacobians_[0] = jacobian_relative_pose;
  jacobians_[1] = jacobian_scale;
}

}  // namespace backend
}  // namespace vins
