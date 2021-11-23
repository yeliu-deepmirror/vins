#pragma once

#include "vins/backend/edge/edge.h"

namespace vins {
namespace backend {

class EdgeMapFusion : public Edge {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeMapFusion(Vec3& pts_i, Vec3& pts_j, const std::vector<std::shared_ptr<Vertex>>& vertices)
      : Edge(3, 2, vertices, std::vector<VertexEdgeTypes>{V_CAMERA_POSE, V_CLOUD_SCALE}),
        pts_i_(pts_i),
        pts_j_(pts_j) {}

  /// return the edge type
  virtual std::string TypeInfo() const override { return "EdgeMapFusion"; }

  virtual void ComputeResidual() override;

  virtual void ComputeJacobians() override;

 private:
  Vec3 pts_i_;
  Vec3 pts_j_;
};

}  // namespace backend
}  // namespace vins
