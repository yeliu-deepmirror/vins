#include "vins/backend/common/integration_base.h"

namespace vins {
namespace backend {

IntegrationBase::IntegrationBase(const Eigen::Vector3d& _acc_0, const Eigen::Vector3d& _gyr_0,
                                 const Eigen::Vector3d& _linearized_ba,
                                 const Eigen::Vector3d& _linearized_bg,
                                 const ImuIntrinsic& intrinsic)
    : acc_0{_acc_0},
      gyr_0{_gyr_0},
      gravity_(Eigen::Vector3d(0, 0, GRAVITY_NORM)),
      linearized_acc{_acc_0},
      linearized_gyr{_gyr_0},
      linearized_ba{_linearized_ba},
      linearized_bg{_linearized_bg},
      jacobian{Eigen::Matrix<double, 15, 15>::Identity()},
      covariance{Eigen::Matrix<double, 15, 15>::Zero()},
      sum_dt{0.0},
      delta_p{Eigen::Vector3d::Zero()},
      delta_q{Eigen::Quaterniond::Identity()},
      delta_v{Eigen::Vector3d::Zero()} {
  noise = Eigen::Matrix<double, 18, 18>::Zero();
  noise.block<3, 3>(0, 0) =
      (intrinsic.acc_noise * intrinsic.acc_noise) * Eigen::Matrix3d::Identity();
  noise.block<3, 3>(3, 3) =
      (intrinsic.gyr_noise * intrinsic.gyr_noise) * Eigen::Matrix3d::Identity();
  noise.block<3, 3>(6, 6) =
      (intrinsic.acc_noise * intrinsic.acc_noise) * Eigen::Matrix3d::Identity();
  noise.block<3, 3>(9, 9) =
      (intrinsic.gyr_noise * intrinsic.gyr_noise) * Eigen::Matrix3d::Identity();
  noise.block<3, 3>(12, 12) =
      (intrinsic.acc_random_walk * intrinsic.acc_random_walk) * Eigen::Matrix3d::Identity();
  noise.block<3, 3>(15, 15) =
      (intrinsic.gyr_random_walk * intrinsic.gyr_random_walk) * Eigen::Matrix3d::Identity();
}

void IntegrationBase::push_back(double dt, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr) {
  dt_buf.push_back(dt);
  acc_buf.push_back(acc);
  gyr_buf.push_back(gyr);
  propagate(dt, acc, gyr);
}

}  // namespace backend
}  // namespace vins
