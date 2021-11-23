#pragma once

#include "vins/backend/edge/edge.h"

namespace vins {
namespace backend {

class EdgeReprojection : public Edge {
 public:
  EdgeReprojection(const Eigen::Vector3d& pts_i, const Eigen::Vector3d& pts_j,
                   const std::vector<std::shared_ptr<Vertex>>& vertices);

  std::string TypeInfo() const override { return "EdgeReprojection"; }
  void ComputeResidual() override;
  void ComputeJacobians() override;

  Eigen::Vector3d GetPointInWorld();

 private:
  Eigen::Vector3d pts_i_, pts_j_;
};

class EdgeReprojectionXYZ : public Edge {
 public:
  EdgeReprojectionXYZ(const Eigen::Vector3d& world_pt,
                      const std::vector<std::shared_ptr<Vertex>>& vertices);

  virtual std::string TypeInfo() const override { return "EdgeReprojectionXYZ"; }
  virtual void ComputeResidual() override;
  virtual void ComputeJacobians() override;

 private:
  Eigen::Vector3d pt_obs_;
};

}  // namespace backend
}  // namespace vins
