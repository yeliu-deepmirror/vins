#pragma once

#include "vins/backend/common/eigen_types.h"

namespace vins {
namespace backend {

class LossFunction {
 public:
  virtual ~LossFunction() {}
  virtual Eigen::Vector3d Compute(double cost) const = 0;
};

class TrivalLoss : public LossFunction {
 public:
  virtual Eigen::Vector3d Compute(double cost) const override { return {cost, 1, 0}; }
};

class HuberLoss : public LossFunction {
 public:
  explicit HuberLoss(double delta) : delta_(delta), sqr_delta_(delta * delta) {}
  virtual Eigen::Vector3d Compute(double cost) const override;

 private:
  double delta_;
  double sqr_delta_;
};

class CauchyLoss : public LossFunction {
 public:
  explicit CauchyLoss(double delta)
      : delta_(delta), sqr_delta_(delta * delta), inv_sqr_delta_(1.0 / sqr_delta_) {}
  virtual Eigen::Vector3d Compute(double cost) const override;

 private:
  double delta_;
  double sqr_delta_;
  double inv_sqr_delta_;
};

class TukeyLoss : public LossFunction {
 public:
  explicit TukeyLoss(double delta) : delta_(delta), sqr_delta_(delta * delta) {}
  virtual Eigen::Vector3d Compute(double cost) const override;

 private:
  double delta_;
  double sqr_delta_;
};

inline double RobustInformation(const LossFunction& loss_fcn, const Eigen::MatrixXd& information,
                                const Eigen::MatrixXd& sqrt_information,
                                const Eigen::VectorXd& residual, double* drho,
                                Eigen::MatrixXd* info) {
  double cost = residual.transpose() * information * residual;
  Eigen::Vector3d rho = loss_fcn.Compute(cost);
  Eigen::VectorXd weight_err = sqrt_information * residual;
  Eigen::MatrixXd robust_info(information.rows(), information.cols());
  robust_info.setIdentity();
  robust_info *= rho[1];
  if (rho[1] + 2.0 * rho[2] * cost > 0.) {
    robust_info += 2.0 * rho[2] * weight_err * weight_err.transpose();
  }
  *info = robust_info * information;
  *drho = rho[1];
  return rho[0];
}

}  // namespace backend
}  // namespace vins
