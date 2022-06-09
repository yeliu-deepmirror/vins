#pragma once

#include "vins/backend/common/utility.h"

namespace vins {
namespace backend {

const double GRAVITY_NORM = 9.81;

struct ImuIntrinsic {
  ImuIntrinsic() {}
  Eigen::Vector3d acc_noise = Eigen::Vector3d::Constant(0.3);
  Eigen::Vector3d gyr_noise = Eigen::Vector3d::Constant(0.06);
  Eigen::Vector3d acc_random_walk = Eigen::Vector3d::Constant(0.04);
  Eigen::Vector3d gyr_random_walk = Eigen::Vector3d::Constant(3e-3);
};

class IntegrationBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  IntegrationBase() = delete;
  IntegrationBase(const Eigen::Vector3d& _acc_0, const Eigen::Vector3d& _gyr_0,
                  const Eigen::Vector3d& _linearized_ba, const Eigen::Vector3d& _linearized_bg,
                  const ImuIntrinsic& intrinsic);

  void push_back(double dt, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr);

  void RePropagate(const Eigen::Vector3d& _linearized_ba, const Eigen::Vector3d& _linearized_bg);

  void MidPointIntegration(double _dt, const Eigen::Vector3d& _acc_0, const Eigen::Vector3d& _gyr_0,
                           const Eigen::Vector3d& _acc_1, const Eigen::Vector3d& _gyr_1,
                           const Eigen::Vector3d& delta_p, const Eigen::Quaterniond& delta_q,
                           const Eigen::Vector3d& delta_v, const Eigen::Vector3d& linearized_ba,
                           const Eigen::Vector3d& linearized_bg, Eigen::Vector3d& result_delta_p,
                           Eigen::Quaterniond& result_delta_q, Eigen::Vector3d& result_delta_v,
                           Eigen::Vector3d& result_linearized_ba,
                           Eigen::Vector3d& result_linearized_bg);

  void Propagate(double _dt, const Eigen::Vector3d& _acc_1, const Eigen::Vector3d& _gyr_1);

  inline Eigen::Matrix<double, 15, 1> Evaluate(
      const Eigen::Vector3d& Pi, const Eigen::Quaterniond& Qi, const Eigen::Vector3d& Vi,
      const Eigen::Vector3d& Bai, const Eigen::Vector3d& Bgi, const Eigen::Vector3d& Pj,
      const Eigen::Quaterniond& Qj, const Eigen::Vector3d& Vj, const Eigen::Vector3d& Baj,
      const Eigen::Vector3d& Bgj) {
    Eigen::Matrix<double, 15, 1> residuals;
    const Eigen::Matrix3d& dp_dba = jacobian.block<3, 3>(0, 9);
    const Eigen::Matrix3d& dp_dbg = jacobian.block<3, 3>(0, 12);
    const Eigen::Matrix3d& dq_dbg = jacobian.block<3, 3>(3, 12);
    const Eigen::Matrix3d& dv_dba = jacobian.block<3, 3>(6, 9);
    const Eigen::Matrix3d& dv_dbg = jacobian.block<3, 3>(6, 12);

    Eigen::Vector3d dba = Bai - linearized_ba;
    Eigen::Vector3d dbg = Bgi - linearized_bg;
    Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
    Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
    Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;
    residuals.block<3, 1>(0, 0) =
        Qi.inverse() * (0.5 * gravity_ * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) -
        corrected_delta_p;
    residuals.block<3, 1>(3, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
    residuals.block<3, 1>(6, 0) = Qi.inverse() * (gravity_ * sum_dt + Vj - Vi) - corrected_delta_v;
    residuals.block<3, 1>(9, 0) = Baj - Bai;
    residuals.block<3, 1>(12, 0) = Bgj - Bgi;
    return residuals;
  }

  double dt;
  Eigen::Vector3d acc_0, gyr_0;
  Eigen::Vector3d acc_1, gyr_1;
  Eigen::Vector3d gravity_;

  const Eigen::Vector3d linearized_acc, linearized_gyr;
  Eigen::Vector3d linearized_ba, linearized_bg;

  Eigen::Matrix<double, 15, 15> jacobian, covariance;
  Eigen::Matrix<double, 15, 15> step_jacobian;
  Eigen::Matrix<double, 15, 18> step_V;
  Eigen::Matrix<double, 18, 18> noise;

  double sum_dt;
  Eigen::Vector3d delta_p;
  Eigen::Quaterniond delta_q;
  Eigen::Vector3d delta_v;

  std::vector<double> dt_buf;
  std::vector<Eigen::Vector3d> acc_buf;
  std::vector<Eigen::Vector3d> gyr_buf;
};

}  // namespace backend
}  // namespace vins
