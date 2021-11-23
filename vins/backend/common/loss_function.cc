#include "vins/backend/common/loss_function.h"

namespace vins {
namespace backend {

Eigen::Vector3d HuberLoss::Compute(double cost) const {
  if (cost <= sqr_delta_) {
    return {cost, 1, 0};
  } else {
    double sqrte = sqrt(cost);
    double tmp = delta_ / sqrte;
    return {2 * sqrte * delta_ - sqr_delta_, tmp, -0.5 * tmp / cost};
  }
}

Eigen::Vector3d CauchyLoss::Compute(double cost) const {
  double aux = inv_sqr_delta_ * cost + 1.0;
  double tmp = 1. / aux;
  return {sqr_delta_ * log(aux), tmp, -inv_sqr_delta_ * std::pow(tmp, 2)};
}

Eigen::Vector3d TukeyLoss::Compute(double cost) const {
  if (sqrt(cost) <= delta_) {
    double aux = cost / sqr_delta_;
    return {sqr_delta_ * (1. - std::pow((1. - aux), 3)) / 3., std::pow((1. - aux), 2),
            -2. * (1. - aux) / sqr_delta_};
  } else {
    return {sqr_delta_ / 3., 0, 0};
  }
}

}  // namespace backend
}  // namespace vins
