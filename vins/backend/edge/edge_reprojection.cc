
#include "vins/backend/edge/edge_reprojection.h"

#include "vins/backend/common/utility.h"
#include "vins/backend/vertex/vertex_pose.h"

namespace vins {
namespace backend {

EdgeReprojection::EdgeReprojection(const Eigen::Vector3d& pts_i, const Eigen::Vector3d& pts_j,
                                   const std::vector<std::shared_ptr<Vertex>>& vertices)
    : Edge(2, 4, vertices,
           std::vector<VertexEdgeTypes>{V_INVERSE_DEPTH, V_CAMERA_POSE, V_CAMERA_POSE,
                                        V_CAMERA_POSE}) {
  pts_i_ = pts_i;
  pts_j_ = pts_j;
}

void EdgeReprojection::ComputeResidual() {
  double inv_dep_i = vertices_[0]->Parameters()[0];

  Eigen::VectorXd param_i = vertices_[1]->Parameters();
  Eigen::Quaterniond Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
  Eigen::Vector3d Pi = param_i.head<3>();

  Eigen::VectorXd param_j = vertices_[2]->Parameters();
  Eigen::Quaterniond Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
  Eigen::Vector3d Pj = param_j.head<3>();

  Eigen::VectorXd param_ext = vertices_[3]->Parameters();
  Eigen::Quaterniond qic(param_ext[6], param_ext[3], param_ext[4], param_ext[5]);
  Eigen::Vector3d tic = param_ext.head<3>();

  Eigen::Vector3d pts_camera_i = pts_i_ / inv_dep_i;
  Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
  Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
  Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
  Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);

  double dep_j = pts_camera_j.z();
  residual_ = (pts_camera_j / dep_j).head<2>() - pts_j_.head<2>();
}

Eigen::Vector3d EdgeReprojection::GetPointInWorld() {
  double inv_dep_i = vertices_[0]->Parameters()[0];

  Eigen::VectorXd param_i = vertices_[1]->Parameters();
  Eigen::Quaterniond Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
  Eigen::Vector3d Pi = param_i.head<3>();

  Eigen::VectorXd param_ext = vertices_[3]->Parameters();
  Eigen::Quaterniond qic(param_ext[6], param_ext[3], param_ext[4], param_ext[5]);
  Eigen::Vector3d tic = param_ext.head<3>();

  Eigen::Vector3d pts_camera_i = pts_i_ / inv_dep_i;
  Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
  Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;

  return pts_w;
}

void EdgeReprojection::ComputeJacobians() {
  double inv_dep_i = vertices_[0]->Parameters()[0];

  Eigen::VectorXd param_i = vertices_[1]->Parameters();
  Eigen::Quaterniond Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
  Eigen::Vector3d Pi = param_i.head<3>();

  Eigen::VectorXd param_j = vertices_[2]->Parameters();
  Eigen::Quaterniond Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
  Eigen::Vector3d Pj = param_j.head<3>();

  Eigen::VectorXd param_ext = vertices_[3]->Parameters();
  Eigen::Quaterniond qic(param_ext[6], param_ext[3], param_ext[4], param_ext[5]);
  Eigen::Vector3d tic = param_ext.head<3>();

  Eigen::Vector3d pts_camera_i = pts_i_ / inv_dep_i;
  Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
  Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
  Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
  Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);

  double dep_j = pts_camera_j.z();

  Mat33 Ri = Qi.toRotationMatrix();
  Mat33 Rj = Qj.toRotationMatrix();
  Mat33 ric = qic.toRotationMatrix();
  Mat23 reduce(2, 3);
  reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j), 0, 1. / dep_j,
      -pts_camera_j(1) / (dep_j * dep_j);

  if (true) {
    Eigen::Matrix<double, 2, 6> jacobian_pose_i;
    Eigen::Matrix<double, 3, 6> jaco_i;
    jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
    jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Sophus::SO3d::hat(pts_imu_i);
    jacobian_pose_i.leftCols<6>() = reduce * jaco_i;

    Eigen::Matrix<double, 2, 6> jacobian_pose_j;
    Eigen::Matrix<double, 3, 6> jaco_j;
    jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
    jaco_j.rightCols<3>() = ric.transpose() * Sophus::SO3d::hat(pts_imu_j);
    jacobian_pose_j.leftCols<6>() = reduce * jaco_j;

    jacobians_[1] = jacobian_pose_i;
    jacobians_[2] = jacobian_pose_j;
  }

  Eigen::Vector2d jacobian_feature;
  jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i_ * -1.0 /
                     (inv_dep_i * inv_dep_i);

  Eigen::Matrix<double, 2, 6> jacobian_ex_pose;
  Eigen::Matrix<double, 3, 6> jaco_ex;
  jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
  Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
  jaco_ex.rightCols<3>() =
      -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
      Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
  jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;

  jacobians_[0] = jacobian_feature;
  jacobians_[3] = jacobian_ex_pose;
}

EdgeReprojectionXYZ::EdgeReprojectionXYZ(const Eigen::Vector3d& world_pt,
                                         const std::vector<std::shared_ptr<Vertex>>& vertices)
    : Edge(2, 2, vertices, std::vector<VertexEdgeTypes>{V_POINT_XYZ, V_CAMERA_POSE}),
      pt_obs_(world_pt) {}

void EdgeReprojectionXYZ::ComputeResidual() {
  Eigen::Vector3d pt_w = vertices_[0]->Parameters();
  Eigen::VectorXd camera_pose = vertices_[1]->Parameters();

  Eigen::Quaterniond rotation(camera_pose[6], camera_pose[3], camera_pose[4], camera_pose[5]);
  Eigen::Vector3d pt_cam = rotation.inverse() * (pt_w - camera_pose.head<3>());
  residual_ = (pt_cam / pt_cam.z()).head<2>() - pt_obs_.head<2>();
}

void EdgeReprojectionXYZ::ComputeJacobians() {
  Eigen::Vector3d pts_w = vertices_[0]->Parameters();

  Eigen::VectorXd param_i = vertices_[1]->Parameters();
  Eigen::Quaterniond Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
  Eigen::Vector3d Pi = param_i.head<3>();

  Eigen::Vector3d pts_camera_i = Qi.inverse() * (pts_w - Pi);

  double dep_i = pts_camera_i.z();

  Mat33 Ri = Qi.toRotationMatrix();

  Mat23 reduce(2, 3);
  reduce << 1. / dep_i, 0, -pts_camera_i(0) / (dep_i * dep_i), 0, 1. / dep_i,
      -pts_camera_i(1) / (dep_i * dep_i);

  Eigen::Matrix<double, 2, 6> jacobian_pose_i;
  Eigen::Matrix<double, 3, 6> jaco_i;
  jaco_i.leftCols<3>() = -Ri.transpose();
  jaco_i.rightCols<3>() = Sophus::SO3d::hat(pts_camera_i);
  jacobian_pose_i.leftCols<6>() = reduce * jaco_i;

  Eigen::Matrix<double, 2, 3> jacobian_feature;
  jacobian_feature = reduce * Ri.transpose();

  jacobians_[0] = jacobian_feature;
  jacobians_[1] = jacobian_pose_i;
}

}  // namespace backend
}  // namespace vins
