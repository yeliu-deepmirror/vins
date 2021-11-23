#pragma once

#include "vins/backend/vertex/vertex.h"

namespace vins {
namespace backend {

class VertexPointXYZ : public Vertex {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  VertexPointXYZ() : Vertex(3) {}

  VertexEdgeTypes TypeId() const { return V_POINT_XYZ; }
};

}  // namespace backend
}  // namespace vins
