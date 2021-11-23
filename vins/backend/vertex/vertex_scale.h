#pragma once

#include "vins/backend/vertex/vertex.h"

namespace vins {
namespace backend {

class VertexScale : public Vertex {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  VertexScale() : Vertex(1) {}

  VertexEdgeTypes TypeId() const { return V_CLOUD_SCALE; }
};

}  // namespace backend
}  // namespace vins
