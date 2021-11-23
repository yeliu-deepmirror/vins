
#include "vins/backend/vertex/vertex_pose.h"

namespace vins {
namespace backend {

void VertexPose::Plus(const Eigen::VectorXd& delta) {
  Eigen::VectorXd& parameters = Parameters();
  parameters.head<3>() += delta.head<3>();
  Eigen::Quaterniond q(parameters[6], parameters[3], parameters[4], parameters[5]);
  q = q * Sophus::SO3d::exp(Vec3(delta[3], delta[4], delta[5])).unit_quaternion();
  q.normalized();
  parameters[3] = q.x();
  parameters[4] = q.y();
  parameters[5] = q.z();
  parameters[6] = q.w();
}

}  // namespace backend
}  // namespace vins
