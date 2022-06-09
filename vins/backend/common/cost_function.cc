#include "vins/backend/common/cost_function.h"

namespace vins {
namespace backend {

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

void ComputeImuJacobian(const Eigen::Vector3d& t_w_i1, const Eigen::Quaterniond& r_w_i1,
                        const Eigen::Vector3d& v1, const Eigen::Vector3d& ba1,
                        const Eigen::Vector3d& bg1, const Eigen::Vector3d& t_w_i2,
                        const Eigen::Quaterniond& r_w_i2, const Eigen::Vector3d& v2,
                        const Eigen::Vector3d& ba2, const Eigen::Vector3d& bg2,
                        IntegrationBase* pre_integration, Eigen::MatrixXd* jaco_p1,
                        Eigen::MatrixXd* jaco_sb1, Eigen::MatrixXd* jaco_p2,
                        Eigen::MatrixXd* jaco_sb2, Eigen::VectorXd* residual) {
  CHECK(pre_integration != nullptr);
  if (residual != nullptr) {
    *residual =
        pre_integration->Evaluate(t_w_i1, r_w_i1, v1, ba1, bg1, t_w_i2, r_w_i2, v2, ba2, bg2);
  }
  if (jaco_p1 == nullptr && jaco_sb1 == nullptr && jaco_p2 == nullptr && jaco_sb2 == nullptr) {
    return;
  }
  double sum_dt = pre_integration->sum_dt;
  Eigen::Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(O_P, O_BA);
  Eigen::Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(O_P, O_BG);
  Eigen::Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
  Eigen::Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(O_V, O_BA);
  Eigen::Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(O_V, O_BG);

  Eigen::Matrix3d rot1_inv = r_w_i1.matrix().transpose();
  Eigen::Quaterniond q2t_q1 = r_w_i2.inverse() * r_w_i1;

  Eigen::Quaterniond corrected_delta_q =
      pre_integration->delta_q * Utility::deltaQ(dq_dbg * (bg1 - pre_integration->linearized_bg));

  if (jaco_p1 != nullptr) {
    *jaco_p1 = Eigen::Matrix<double, 15, 6>::Zero();

    jaco_p1->block<3, 3>(O_P, O_P) = -rot1_inv;
    jaco_p1->block<3, 3>(O_P, O_R) =
        Utility::skewSymmetric(rot1_inv * (0.5 * pre_integration->gravity_ * sum_dt * sum_dt +
                                           t_w_i2 - t_w_i1 - v1 * sum_dt));
    jaco_p1->block<3, 3>(O_R, O_R) =
        -(Utility::Qleft(q2t_q1) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
    jaco_p1->block<3, 3>(O_V, O_R) =
        Utility::skewSymmetric(rot1_inv * (pre_integration->gravity_ * sum_dt + v2 - v1));
  }

  if (jaco_sb1 != nullptr) {
    *jaco_sb1 = Eigen::Matrix<double, 15, 9>::Zero();
    jaco_sb1->block<3, 3>(O_P, O_V - O_V) = -rot1_inv * sum_dt;
    jaco_sb1->block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
    jaco_sb1->block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;

    jaco_sb1->block<3, 3>(O_R, O_BG - O_V) =
        -Utility::Qleft(q2t_q1 * pre_integration->delta_q).bottomRightCorner<3, 3>() * dq_dbg;

    jaco_sb1->block<3, 3>(O_V, O_V - O_V) = -rot1_inv;
    jaco_sb1->block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
    jaco_sb1->block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;

    jaco_sb1->block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();
    jaco_sb1->block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();
  }

  if (jaco_p2 != nullptr) {
    *jaco_p2 = Eigen::Matrix<double, 15, 6>::Zero();
    jaco_p2->block<3, 3>(O_P, O_P) = rot1_inv;
    jaco_p2->block<3, 3>(O_R, O_R) =
        Utility::Qleft(corrected_delta_q.inverse() * r_w_i1.inverse() * r_w_i2)
            .bottomRightCorner<3, 3>();
  }

  if (jaco_sb2 != nullptr) {
    *jaco_sb2 = Eigen::Matrix<double, 15, 9>::Zero();
    jaco_sb2->block<3, 3>(O_V, O_V - O_V) = rot1_inv;
    jaco_sb2->block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();
    jaco_sb2->block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();
  }
}

}  // namespace backend
}  // namespace vins
