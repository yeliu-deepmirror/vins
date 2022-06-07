#pragma once

#include "vins/backend/common/integration_base.h"
#include "vins/backend/edge/edge.h"

namespace vins {
namespace backend {

class EdgeImu : public Edge {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  explicit EdgeImu(IntegrationBase* _pre_integration,
                   const std::vector<std::shared_ptr<Vertex>>& vertices)
      : Edge(15, 4, vertices,
             std::vector<VertexEdgeTypes>{V_CAMERA_POSE, V_IMU_SPEED_BIAS, V_CAMERA_POSE,
                                          V_IMU_SPEED_BIAS}),
        pre_integration_(_pre_integration) {}

  virtual std::string TypeInfo() const override { return "EdgeImu"; }
  virtual void ComputeResidual() override;
  virtual void ComputeJacobians() override;

 private:
  enum StateOrder { O_P = 0, O_R = 3, O_V = 6, O_BA = 9, O_BG = 12 };
  IntegrationBase* pre_integration_;
  static Vec3 gravity_;

  Mat33 dp_dba_ = Mat33::Zero();
  Mat33 dp_dbg_ = Mat33::Zero();
  Mat33 dr_dbg_ = Mat33::Zero();
  Mat33 dv_dba_ = Mat33::Zero();
  Mat33 dv_dbg_ = Mat33::Zero();
};

void ComputeImuJacobian(const Eigen::Vector3d& t_w_i1, const Eigen::Quaterniond& r_w_i1,
                        const Eigen::Vector3d& v1, const Eigen::Vector3d& ba1,
                        const Eigen::Vector3d& bg1, const Eigen::Vector3d& t_w_i2,
                        const Eigen::Quaterniond& r_w_i2, const Eigen::Vector3d& v2,
                        const Eigen::Vector3d& ba2, const Eigen::Vector3d& bg2,
                        IntegrationBase* pre_integration, Eigen::MatrixXd* jaco_p1,
                        Eigen::MatrixXd* jaco_sb1, Eigen::MatrixXd* jaco_p2,
                        Eigen::MatrixXd* jaco_sb2, Eigen::VectorXd* residual);

}  // namespace backend
}  // namespace vins
