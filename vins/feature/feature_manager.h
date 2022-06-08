#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <vector>

using namespace std;

#include <Eigen/Dense>

namespace vins {
namespace feature {

const int NUM_OF_CAM = 1;
const int WINDOW_SIZE = 10;
const double MIN_PARALLAX = 0.04;
const double INIT_DEPTH = 2.0;

struct FeaturePerFrame {
  FeaturePerFrame() = default;
  explicit FeaturePerFrame(const Eigen::Matrix<double, 3, 1>& _point) : point(_point) {
    if (point(2) < 0.1) point(2) = 1.0;
  }
  Eigen::Vector3d point;
};

class FeaturePerId {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  const uint64_t feature_id;
  int start_frame;
  std::vector<FeaturePerFrame> feature_per_frame;

  double estimated_depth;
  std::optional<double> depth_gt;
  // 0 haven't solve yet; 1 solve succ; 2 solve fail; 3 good initial
  // we good initial we will skip optimization & triangulation
  int solve_flag;

  FeaturePerId(int _feature_id, int _start_frame)
      : feature_id(_feature_id), start_frame(_start_frame), estimated_depth(-1.0), solve_flag(0) {}

  int endFrame();

  bool Valid() { return (feature_per_frame.size() >= 2 && start_frame < WINDOW_SIZE - 2); }
};

class FeatureManager {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  FeatureManager(Eigen::Matrix3d _Rs[]);

  void ClearState();

  bool AddFeatureCheckParallax(
      int frame_count,
      const std::map<uint64_t, std::vector<std::pair<int, Eigen::Vector3d>>>& image);
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> GetCorresponding(int frame_count_l,
                                                                            int frame_count_r);

  void removeFailures();
  void ClearDepth();
  void Triangulate(Eigen::Vector3d Ps[], const Eigen::Vector3d& tic, const Eigen::Matrix3d& ric);
  void RemoveBackShiftDepth(const Eigen::Matrix3d& marg_R, const Eigen::Vector3d& marg_P,
                            const Eigen::Matrix3d& new_R, const Eigen::Vector3d& new_P);
  void RemoveBack();
  void removeFront(int frame_count);

  std::list<FeaturePerId> feature;
  int last_track_num;

 private:
  double CompensatedParallax(const FeaturePerId& it_per_id, int frame_count);
  const Eigen::Matrix3d* Rs;
};

}  // namespace feature
}  // namespace vins

#endif
