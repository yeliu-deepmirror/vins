#pragma once

#include "vins/backend/edge/edge.h"

namespace vins {
namespace backend {

class EdgeSE3Prior : public Edge {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeSE3Prior(const Vec3& p, const Qd& q, const std::vector<std::shared_ptr<Vertex>>& vertices)
      : Edge(6, 1, vertices, std::vector<VertexEdgeTypes>{V_CAMERA_POSE}), Pp_(p), Qp_(q) {}

  virtual std::string TypeInfo() const override { return "EdgeSE3Prior"; }

  virtual void ComputeResidual() override;

  virtual void ComputeJacobians() override;

 private:
  Vec3 Pp_;  // pose prior
  Qd Qp_;    // Rotation prior
};

}  // namespace backend
}  // namespace vins
