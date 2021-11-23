#pragma once

#include "vins/backend/vertex/vertex.h"

namespace vins {
namespace backend {

class VertexInverseDepth : public Vertex {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  VertexInverseDepth() : Vertex(1) {}

  VertexEdgeTypes TypeId() const { return V_INVERSE_DEPTH; }
};

}  // namespace backend
}  // namespace vins
