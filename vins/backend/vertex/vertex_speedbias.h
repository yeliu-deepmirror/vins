#pragma once

#include "vins/backend/vertex/vertex.h"

namespace vins {
namespace backend {

class VertexSpeedBias : public Vertex {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  VertexSpeedBias() : Vertex(9) {}

  VertexEdgeTypes TypeId() const { return V_IMU_SPEED_BIAS; }
};

}  // namespace backend
}  // namespace vins
