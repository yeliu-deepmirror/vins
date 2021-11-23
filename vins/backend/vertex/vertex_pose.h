#pragma once

#include "vins/backend/vertex/vertex.h"

namespace vins {
namespace backend {

class VertexPose : public Vertex {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  VertexPose() : Vertex(7, 6) {}

  virtual void Plus(const VecX& delta) override;
  VertexEdgeTypes TypeId() const { return V_CAMERA_POSE; }
};

}  // namespace backend
}  // namespace vins
