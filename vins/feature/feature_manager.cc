
#include "vins/feature/feature_manager.h"

namespace vins {
namespace feature {

int FeaturePerId::endFrame() { return start_frame + feature_per_frame.size() - 1; }

FeatureManager::FeatureManager(Eigen::Matrix3d _Rs[]) : Rs(_Rs) {}

void FeatureManager::ClearState() { feature.clear(); }

int FeatureManager::GetFeatureCount() {
  int cnt = 0;
  for (auto& it : feature) {
    it.used_num = it.feature_per_frame.size();
    if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2) cnt++;
  }
  return cnt;
}

bool FeatureManager::AddFeatureCheckParallax(
    int frame_count, const std::map<int, std::vector<std::pair<int, Eigen::Vector3d>>>& image,
    double) {
  last_track_num = 0;
  for (auto& id_pts : image) {
    const Eigen::Vector3d& pt_cam = id_pts.second[0].second;
    int feature_id = id_pts.first;
    std::list<FeaturePerId>::iterator it =
        find_if(feature.begin(), feature.end(),
                [feature_id](const FeaturePerId& it) { return it.feature_id == feature_id; });
    if (it == feature.end()) {
      feature.push_back(FeaturePerId(feature_id, frame_count));
      it = std::prev(feature.end());
      if (pt_cam(2) > 0.1) {
        // we have good depth initialization (maybe from other sensor)
        it->estimated_depth = pt_cam(2);
        it->solve_flag = 3;
        LOG(INFO) << "have initial depth";
      }
    } else {
      last_track_num++;
    }
    it->feature_per_frame.emplace_back(FeaturePerFrame(pt_cam));
  }

  if (frame_count < 2 || last_track_num < 20) return true;

  double parallax_sum = 0;
  int parallax_num = 0;
  for (auto& it_per_id : feature) {
    if (it_per_id.start_frame <= frame_count - 2 &&
        it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1) {
      parallax_sum += CompensatedParallax(it_per_id, frame_count);
      parallax_num++;
    }
  }

  if (parallax_num == 0) return true;
  return parallax_sum / parallax_num >= MIN_PARALLAX;
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> FeatureManager::GetCorresponding(
    int frame_count_l, int frame_count_r) {
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
  for (auto& it : feature) {
    if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r) {
      int idx_l = frame_count_l - it.start_frame;
      int idx_r = frame_count_r - it.start_frame;
      corres.push_back(
          std::make_pair(it.feature_per_frame[idx_l].point, it.feature_per_frame[idx_r].point));
    }
  }
  return corres;
}

void FeatureManager::SetDepth(const Eigen::VectorXd& x) {
  int feature_index = 0;
  for (auto& it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) continue;
    double depth = 1.0 / x(feature_index++);

    if (it_per_id.solve_flag == 3) continue;

    it_per_id.estimated_depth = depth;
    if (it_per_id.estimated_depth < 0) {
      it_per_id.solve_flag = 2;
    } else {
      it_per_id.solve_flag = 1;
    }
  }
}

void FeatureManager::removeFailures() {
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
    it_next++;
    if (it->solve_flag == 2) feature.erase(it);
  }
}

void FeatureManager::ClearDepth() {
  for (auto& it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) continue;
    it_per_id.estimated_depth = -1;
  }
}

Eigen::VectorXd FeatureManager::GetInverseDepthVector() {
  Eigen::VectorXd dep_vec(GetFeatureCount());
  int feature_index = -1;
  for (auto& it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) continue;

    dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
  }
  return dep_vec;
}

void FeatureManager::triangulate(Eigen::Vector3d Ps[], Eigen::Vector3d tic,
                                 Eigen::Matrix3d ric) {
  for (auto& it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) continue;

    if (it_per_id.estimated_depth > 0) continue;
    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

    assert(NUM_OF_CAM == 1);
    Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
    int svd_idx = 0;

    Eigen::Matrix<double, 3, 4> P0;
    Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic;
    Eigen::Matrix3d R0 = Rs[imu_i] * ric;
    P0.leftCols<3>() = Eigen::Matrix3d::Identity();
    P0.rightCols<1>() = Eigen::Vector3d::Zero();

    for (auto& it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;

      Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic;
      Eigen::Matrix3d R1 = Rs[imu_j] * ric;
      Eigen::Vector3d t = R0.transpose() * (t1 - t0);
      Eigen::Matrix3d R = R0.transpose() * R1;
      Eigen::Matrix<double, 3, 4> P;
      P.leftCols<3>() = R.transpose();
      P.rightCols<1>() = -R.transpose() * t;
      Eigen::Vector3d f = it_per_frame.point.normalized();
      svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
      svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

      if (imu_i == imu_j) continue;
    }
    assert(svd_idx == svd_A.rows());
    Eigen::Vector4d svd_V =
        Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
    it_per_id.estimated_depth = svd_V[2] / svd_V[3];
    if (it_per_id.estimated_depth < 0.1) {
      it_per_id.estimated_depth = INIT_DEPTH;
    }
  }
}

void FeatureManager::RemoveBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P,
                                          Eigen::Matrix3d new_R, Eigen::Vector3d new_P) {
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
    it_next++;
    if (it->start_frame != 0) {
      it->start_frame--;
      continue;
    }

    const Eigen::Vector3d& uv_i = it->feature_per_frame[0].point;
    it->feature_per_frame.erase(it->feature_per_frame.begin());
    if (it->feature_per_frame.size() < 2) {
      feature.erase(it);
      continue;
    } else {
      Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
      Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
      Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
      it->estimated_depth = pts_j(2) > 0.1 ? pts_j(2) : INIT_DEPTH;
    }
  }
}

void FeatureManager::RemoveBack() {
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame != 0)
      it->start_frame--;
    else {
      it->feature_per_frame.erase(it->feature_per_frame.begin());
      if (it->feature_per_frame.size() == 0) feature.erase(it);
    }
  }
}

void FeatureManager::removeFront(int frame_count) {
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
    it_next++;
    if (it->start_frame == frame_count) {
      it->start_frame--;
    } else {
      int j = WINDOW_SIZE - 1 - it->start_frame;
      if (it->endFrame() < frame_count - 1) continue;
      it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
      if (it->feature_per_frame.size() == 0) feature.erase(it);
    }
  }
}

double FeatureManager::CompensatedParallax(const FeaturePerId& it_per_id, int frame_count) {
  // check the second last frame is keyframe or not
  // parallax between seconde last frame and third last frame
  const FeaturePerFrame& frame_i =
      it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
  const FeaturePerFrame& frame_j =
      it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

  Eigen::Vector3d p_j = frame_j.point;

  double u_j = p_j(0);
  double v_j = p_j(1);

  Eigen::Vector3d p_i = frame_i.point;
  Eigen::Vector3d p_i_comp;

  double dep_i = p_i(2);
  double u_i = p_i(0) / dep_i;
  double v_i = p_i(1) / dep_i;
  double du = u_i - u_j, dv = v_i - v_j;
  double ans = sqrt(du * du + dv * dv);

  return ans;
}

}  // namespace feature
}  // namespace vins
