#pragma once

#include "vins/backend/common/eigen_types.h"

namespace vins {
namespace backend {

extern unsigned long global_vertex_id;

class Vertex {
 public:
  explicit Vertex(int num_dimension, int local_dimension = -1);
  virtual ~Vertex();

  int Dimension() const;
  int LocalDimension() const;

  unsigned long Id() const { return id_; }

  void SetParameters(const Eigen::VectorXd& params) { parameters_ = params; }
  Eigen::VectorXd Parameters() const { return parameters_; }
  Eigen::VectorXd& Parameters() { return parameters_; }

  void BackUpParameters() { parameters_backup_ = parameters_; }
  void RollBackParameters() { parameters_ = parameters_backup_; }

  /// add the changement to parameters
  virtual void Plus(const Eigen::VectorXd& delta);
  virtual VertexEdgeTypes TypeId() const = 0;

  // reset ordering
  int OrderingId() const { return ordering_id_; }
  void SetOrderingId(unsigned long id) { ordering_id_ = id; };

  // whether this vertex is fixed or not
  void SetFixed(bool fixed = true) { fixed_ = fixed; }
  bool IsFixed() const { return fixed_; }

 protected:
  Eigen::VectorXd parameters_;
  Eigen::VectorXd parameters_backup_;
  int local_dimension_;

  // vertex id, will be initialized automaticly
  unsigned long id_;

  unsigned long ordering_id_ = 0;
  bool fixed_ = false;
};

}  // namespace backend
}  // namespace vins
