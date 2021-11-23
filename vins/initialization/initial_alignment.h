#pragma once

#include "vins/backend/common/integration_base.h"
#include "vins/backend/common/utility.h"
#include "vins/feature/feature_manager.h"

using namespace Eigen;
using namespace std;

namespace vins {
namespace backend {

class ImageFrame {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  ImageFrame(){};
  ImageFrame(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>& _points,
             double _t)
      : t{_t}, is_key_frame{false} {
    points = _points;
  };
  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> points;
  double t;
  Eigen::Matrix3d R;
  Eigen::Vector3d T;
  IntegrationBase* pre_integration;
  bool is_key_frame;
};

bool VisualIMUAlignment(const Eigen::Vector3d& trans_ic,
                        std::map<double, ImageFrame>& all_image_frame, Eigen::Vector3d* Bgs,
                        Eigen::Vector3d& g, Eigen::VectorXd& x);

}  // namespace backend
}  // namespace vins
