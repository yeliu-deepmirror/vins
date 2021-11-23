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

}  // namespace backend
}  // namespace vins
