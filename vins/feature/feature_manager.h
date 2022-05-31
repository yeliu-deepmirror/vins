#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

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

class FeaturePerFrame {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  FeaturePerFrame(const Eigen::Matrix<double, 7, 1>& _point, double td) {
    point.x() = _point(0);
    point.y() = _point(1);
    point.z() = _point(2);
    uv.x() = _point(3);
    uv.y() = _point(4);
    velocity.x() = _point(5);
    velocity.y() = _point(6);
    cur_td = td;
  }
  double cur_td;
  Eigen::Vector3d point;
  Eigen::Vector2d uv;
  Eigen::Vector2d velocity;
  double z = INIT_DEPTH;
  bool is_used = false;
  double parallax = 0.0;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  double dep_gradient = 0.0;
};

class FeaturePerId {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  const int feature_id;
  int start_frame;
  std::vector<FeaturePerFrame> feature_per_frame;

  int used_num;
  bool is_outlier;
  bool is_margin;
  double estimated_depth;
  int solve_flag;  // 0 haven't solve yet; 1 solve succ; 2 solve fail;

  Eigen::Vector3d gt_p;

  FeaturePerId(int _feature_id, int _start_frame)
      : feature_id(_feature_id),
        start_frame(_start_frame),
        used_num(0),
        estimated_depth(-1.0),
        solve_flag(0) {}

  int endFrame();
};

class FeatureManager {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  FeatureManager(Eigen::Matrix3d _Rs[]);

  void ClearState();

  int GetFeatureCount();

  bool AddFeatureCheckParallax(
      int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& image,
      double td);
  vector<pair<Eigen::Vector3d, Eigen::Vector3d>> GetCorresponding(int frame_count_l,
                                                                  int frame_count_r);

  void SetDepth(const Eigen::VectorXd& x);
  void removeFailures();
  void ClearDepth();
  Eigen::VectorXd GetInverseDepthVector();
  void triangulate(Eigen::Vector3d Ps[], Eigen::Vector3d tic, Eigen::Matrix3d ric);
  void RemoveBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R,
                            Eigen::Vector3d new_P);
  void RemoveBack();
  void removeFront(int frame_count);
  void removeOutlier();

  list<FeaturePerId> feature;
  int last_track_num;

 private:
  double CompensatedParallax(const FeaturePerId& it_per_id, int frame_count);
  const Eigen::Matrix3d* Rs;
};

}  // namespace feature
}  // namespace vins

#endif
