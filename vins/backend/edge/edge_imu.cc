#include "vins/backend/edge/edge_imu.h"
#include "vins/backend/vertex/vertex_pose.h"
#include "vins/backend/vertex/vertex_speedbias.h"

namespace vins {
namespace backend {
using Sophus::SO3d;

Vec3 EdgeImu::gravity_ = Vec3(0, 0, 9.8);

void EdgeImu::ComputeResidual() {
  VecX param_0 = vertices_[0]->Parameters();
  Qd Qi(param_0[6], param_0[3], param_0[4], param_0[5]);
  Vec3 Pi = param_0.head<3>();

  VecX param_1 = vertices_[1]->Parameters();
  Vec3 Vi = param_1.head<3>();
  Vec3 Bai = param_1.segment(3, 3);
  Vec3 Bgi = param_1.tail<3>();

  VecX param_2 = vertices_[2]->Parameters();
  Qd Qj(param_2[6], param_2[3], param_2[4], param_2[5]);
  Vec3 Pj = param_2.head<3>();

  VecX param_3 = vertices_[3]->Parameters();
  Vec3 Vj = param_3.head<3>();
  Vec3 Baj = param_3.segment(3, 3);
  Vec3 Bgj = param_3.tail<3>();

  residual_ = pre_integration_->evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);
  SetInformation(pre_integration_->covariance.inverse());
}

// void ComputeImuResidual(
//     const Eigen::Vector3d& t_w_i1, const Eigen::Quaterniond& r_w_i1,
//     const Eigen::Vector3d& v1, const Eigen::Vector3d& ba1, const Eigen::Vector3d& bg1,
//     const Eigen::Vector3d& t_w_i2, const Eigen::Quaterniond& r_w_i2,
//     const Eigen::Vector3d& v2, const Eigen::Vector3d& ba2, const Eigen::Vector3d& bg2,
//     IntegrationBase* pre_integration, Eigen::VectorXd* residual, Eigen::MatrixXd* information) {
//   *residual_ = pre_integration->evaluate(
//       t_w_i1, r_w_i1, v1, ba1, bg1, t_w_i2, r_w_i2, v2, ba2, bg2);
//   *information = pre_integration_->covariance.inverse();
// }

void EdgeImu::ComputeJacobians() {
  VecX param_0 = vertices_[0]->Parameters();
  Qd Qi(param_0[6], param_0[3], param_0[4], param_0[5]);
  Vec3 Pi = param_0.head<3>();

  VecX param_1 = vertices_[1]->Parameters();
  Vec3 Vi = param_1.head<3>();
  Vec3 Bai = param_1.segment(3, 3);
  Vec3 Bgi = param_1.tail<3>();

  VecX param_2 = vertices_[2]->Parameters();
  Qd Qj(param_2[6], param_2[3], param_2[4], param_2[5]);
  Vec3 Pj = param_2.head<3>();

  VecX param_3 = vertices_[3]->Parameters();
  Vec3 Vj = param_3.head<3>();
  Vec3 Baj = param_3.segment(3, 3);
  Vec3 Bgj = param_3.tail<3>();

  ComputeImuJacobian(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj, pre_integration_, &(jacobians_[0]),
                     &(jacobians_[1]), &(jacobians_[2]), &(jacobians_[3]), nullptr);
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
        pre_integration->evaluate(t_w_i1, r_w_i1, v1, ba1, bg1, t_w_i2, r_w_i2, v2, ba2, bg2);
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
