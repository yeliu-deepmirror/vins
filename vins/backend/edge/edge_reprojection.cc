
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

  residual_ = ComputeVisualResidual(tic, qic.matrix(), Pi, Qi.matrix(), Pj, Qj.matrix(), inv_dep_i,
                                    pts_i_, pts_j_);
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

  ComputeVisualJacobian(tic, qic.matrix(), Pi, Qi.matrix(), Pj, Qj.matrix(), inv_dep_i, pts_i_,
                        pts_j_, &(jacobians_[1]), &(jacobians_[2]), &(jacobians_[3]),
                        &(jacobians_[0]));
}

Eigen::Vector2d ComputeVisualResidual(const Eigen::Vector3d& t_i_c, const Eigen::Matrix3d& r_i_c,
                                      const Eigen::Vector3d& t_w_i1, const Eigen::Matrix3d& r_w_i1,
                                      const Eigen::Vector3d& t_w_i2, const Eigen::Matrix3d& r_w_i2,
                                      double inv_dep_1, const Eigen::Vector3d& pt1,
                                      const Eigen::Vector3d& pt2) {
  Eigen::Vector3d pt_camera_1 = pt1 / inv_dep_1;
  Eigen::Vector3d pt_w = r_w_i1 * (r_i_c * pt_camera_1 + t_i_c) + t_w_i1;
  Eigen::Vector3d pt_imu_2 = r_w_i2.transpose() * (pt_w - t_w_i2);
  Eigen::Vector3d pt_camera_2 = r_i_c.transpose() * (pt_imu_2 - t_i_c);

  double dep_2 = pt_camera_2.z();
  return (pt_camera_2 / dep_2).head<2>() - pt2.head<2>();
}

void ComputeVisualJacobian(const Eigen::Vector3d& t_i_c, const Eigen::Matrix3d& r_i_c,
                           const Eigen::Vector3d& t_w_i1, const Eigen::Matrix3d& r_w_i1,
                           const Eigen::Vector3d& t_w_i2, const Eigen::Matrix3d& r_w_i2,
                           double inv_dep_1, const Eigen::Vector3d& pt1, const Eigen::Vector3d& pt2,
                           Eigen::MatrixXd* jaco_1, Eigen::MatrixXd* jaco_2,
                           Eigen::MatrixXd* jaco_ex, Eigen::MatrixXd* jaco_pt,
                           Eigen::VectorXd* residual) {
  Eigen::Vector3d pt_camera_1 = pt1 / inv_dep_1;
  Eigen::Vector3d pts_imu_1 = r_i_c * pt_camera_1 + t_i_c;
  Eigen::Vector3d pt_w = r_w_i1 * pts_imu_1 + t_w_i1;
  Eigen::Vector3d pt_imu_2 = r_w_i2.transpose() * (pt_w - t_w_i2);
  Eigen::Vector3d pt_camera_2 = r_i_c.transpose() * (pt_imu_2 - t_i_c);

  double inv_dep_2 = 1.0 / pt_camera_2.z();
  double sqr_inv_dep_2 = inv_dep_2 * inv_dep_2;

  if (residual != nullptr) {
    *residual = (pt_camera_2 * inv_dep_2).head<2>() - pt2.head<2>();
  }

  Eigen::Matrix<double, 2, 3> reduce_matrix;
  reduce_matrix << inv_dep_2, 0, -pt_camera_2(0) * sqr_inv_dep_2, 0, inv_dep_2,
      -pt_camera_2(1) * sqr_inv_dep_2;

  Eigen::Matrix3d tmp_1 = Sophus::SO3d::hat(pts_imu_1);
  Eigen::Matrix3d tmp_2 = r_i_c.transpose() * r_w_i2.transpose();

  if (jaco_1 != nullptr) {
    Eigen::Matrix<double, 3, 6> jaco_t1;
    jaco_t1.leftCols<3>() = tmp_2;
    jaco_t1.rightCols<3>() = -tmp_2 * r_w_i1 * tmp_1;
    *jaco_1 = reduce_matrix * jaco_t1;
  }

  if (jaco_2 != nullptr) {
    Eigen::Matrix<double, 3, 6> jaco_t2;
    jaco_t2.leftCols<3>() = -tmp_2;
    jaco_t2.rightCols<3>() = r_i_c.transpose() * tmp_1;
    *jaco_2 = reduce_matrix * jaco_t2;
  }

  if (jaco_pt != nullptr) {
    *jaco_pt = reduce_matrix * tmp_2 * r_w_i1 * r_i_c * pt1 * -1.0 / (inv_dep_1 * inv_dep_1);
  }

  if (jaco_ex != nullptr) {
    Eigen::Matrix<double, 3, 6> jaco_tex;
    Eigen::Matrix3d tmp_3 = tmp_2 * r_w_i1;
    jaco_tex.leftCols<3>() = tmp_3 - r_i_c.transpose();
    Eigen::Matrix3d tmp_r = tmp_3 * r_i_c;
    jaco_tex.rightCols<3>() =
        -tmp_r * Utility::skewSymmetric(pt_camera_1) + Utility::skewSymmetric(tmp_r * pt_camera_1) +
        Utility::skewSymmetric(r_i_c.transpose() *
                               (r_w_i2.transpose() * (r_w_i1 * t_i_c + t_w_i1 - t_w_i2) - t_i_c));
    *jaco_ex = reduce_matrix * jaco_tex;
  }
}

}  // namespace backend
}  // namespace vins
