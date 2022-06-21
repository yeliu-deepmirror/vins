#pragma once

#include "vins/backend/common/integration_base.h"
#include "vins/backend/common/utility.h"
#include "vins/feature/feature_manager.h"

using namespace Eigen;
using namespace std;

namespace vins {
namespace backend {

//  additional information from lidar odometry
struct ImuState {
  // camera information
  Sophus::SE3d pose;
  Eigen::Vector3d velocity;
  Eigen::Vector3d bias_acc;
  Eigen::Vector3d bias_gyr;
  Eigen::Vector3d gravity;
};

class ImageFrame {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  ImageFrame(){};
  ImageFrame(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 3, 1>>>>& _points,
             int64_t _t)
      : t{_t}, is_key_frame{false} {
    points = _points;
  };
  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 3, 1>>>> points;
  int64_t t;
  Eigen::Matrix3d R;
  Eigen::Vector3d T;
  IntegrationBase* pre_integration;
  bool is_key_frame;
  std::optional<ImuState> imu_state_ = std::nullopt;
};

bool VisualIMUAlignment(const Eigen::Vector3d& trans_ic,
                        std::map<int64_t, ImageFrame>& all_image_frame, Eigen::Vector3d* Bgs,
                        Eigen::Vector3d& g, Eigen::VectorXd& x);

}  // namespace backend
}  // namespace vins
