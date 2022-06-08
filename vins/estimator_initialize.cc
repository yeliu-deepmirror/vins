
#include "vins/estimator.h"

namespace vins {

bool Estimator::InitialStructure() {
  // check imu observibility
  {
    std::map<int64_t, backend::ImageFrame>::iterator frame_it;
    Eigen::Vector3d sum_g = Vector3d::Zero();
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end();
         frame_it++) {
      double dt = frame_it->second.pre_integration->sum_dt;
      sum_g += frame_it->second.pre_integration->delta_v / dt;
    }
    Vector3d aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
    double var = 0;
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end();
         frame_it++) {
      double dt = frame_it->second.pre_integration->sum_dt;
      Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
      var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
    }
    if (sqrt(var / ((int)all_image_frame.size() - 1)) < 0.25) return false;
  }
  // global sfm
  Quaterniond Q[frame_count + 1];
  Vector3d T[frame_count + 1];
  std::map<int, Eigen::Vector3d> sfm_tracked_points;
  std::vector<SFMFeature> sfm_f;
  for (auto& it_per_id : f_manager.feature) {
    int imu_j = it_per_id.start_frame - 1;
    SFMFeature tmp_feature;
    tmp_feature.state = false;
    tmp_feature.id = it_per_id.feature_id;
    for (auto& it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      Vector3d pts_j = it_per_frame.point;
      tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
    }
    sfm_f.push_back(tmp_feature);
  }
  Matrix3d relative_R;
  Vector3d relative_T;
  int l;
  if (!relativePose(relative_R, relative_T, l)) {
    LOG_IF(WARNING, verbose_) << "[VINS] Not enough features or parallax; Move device around";
    return false;
  }
  GlobalSFM sfm;
  if (!sfm.construct(frame_count + 1, Q, T, l, relative_R, relative_T, sfm_f, sfm_tracked_points)) {
    LOG_IF(WARNING, verbose_) << "[VINS] global SFM failed!";
    marginalization_flag = MARGIN_OLD;
    return false;
  }

  // solve pnp for all frame
  std::map<int64_t, backend::ImageFrame>::iterator frame_it = all_image_frame.begin();
  for (int i = 0; frame_it != all_image_frame.end(); frame_it++) {
    // provide initial guess
    cv::Mat r, rvec, t, D, tmp_r;
    if ((frame_it->first) == Headers[i]) {
      frame_it->second.R = Q[i].toRotationMatrix() * rigid_ic_.so3().matrix().transpose();
      frame_it->second.T = T[i];
      i++;
      continue;
    }
    if ((frame_it->first) > Headers[i]) {
      i++;
    }
    Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
    Vector3d P_inital = -R_inital * T[i];
    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    vector<cv::Point3f> pts_3_vector;
    vector<cv::Point2f> pts_2_vector;
    for (auto& id_pts : frame_it->second.points) {
      for (auto& i_p : id_pts.second) {
        auto it = sfm_tracked_points.find(id_pts.first);
        if (it == sfm_tracked_points.end()) continue;
        const auto& world_pts = it->second;
        pts_3_vector.emplace_back(cv::Point3f(world_pts(0), world_pts(1), world_pts(2)));
        const auto& img_pts = i_p.second.head<2>();
        pts_2_vector.emplace_back(cv::Point2f(img_pts(0), img_pts(1)));
      }
    }
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    if (pts_3_vector.size() < 6) {
      LOG(WARNING) << "Not enough points for solve pnp pts_3_vector size " << pts_3_vector.size();
      return false;
    }
    if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1)) {
      LOG(WARNING) << " solve pnp fail!";
      return false;
    }
    cv::Rodrigues(rvec, r);
    MatrixXd R_pnp, tmp_R_pnp;
    cv::cv2eigen(r, tmp_R_pnp);
    R_pnp = tmp_R_pnp.transpose();
    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    T_pnp = R_pnp * (-T_pnp);
    frame_it->second.R = R_pnp * rigid_ic_.so3().matrix().transpose();
    frame_it->second.T = T_pnp;
  }
  if (VisualInitialAlign())
    return true;
  else {
    cout << "misalign visual structure with IMU" << endl;
    return false;
  }
}

bool Estimator::VisualInitialAlign() {
  CHECK_EQ(frame_count, feature::WINDOW_SIZE);
  Eigen::VectorXd x;
  // solve scale
  if (!VisualIMUAlignment(rigid_ic_.translation(), all_image_frame, Bgs, g, x)) {
    return false;
  }

  double scale = (x.tail<1>())(0);

  // change state
  for (int i = 0; i <= frame_count; i++) {
    Ps[i] = all_image_frame[Headers[i]].T;
    Rs[i] = all_image_frame[Headers[i]].R;
    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
  }

  f_manager.ClearDepth();
  f_manager.Triangulate(Ps, rigid_ic_.translation(), rigid_ic_.so3().matrix());

  Eigen::Vector3d offset_trans = scale * Ps[0] - Rs[0] * rigid_ic_.translation();
  for (int i = 0; i <= frame_count; i++) {
    Ps[i] = scale * Ps[i] - Rs[i] * rigid_ic_.translation() - offset_trans;
    auto iter = all_image_frame.find(Headers[i]);
    CHECK(iter != all_image_frame.end());
    Vs[i] = iter->second.R * x.segment<3>(i * 3);
  }

  for (auto& it_per_id : f_manager.feature) {
    if (!it_per_id.Valid()) continue;
    it_per_id.estimated_depth *= scale;
  }

  // make gravity pointing down
  Matrix3d g_offset = backend::Utility::g2R(g);
  double yaw = backend::Utility::R2ypr(g_offset * Rs[0]).x();
  g_offset = backend::Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * g_offset;
  g = g_offset * g;
  for (int i = 0; i <= frame_count; i++) {
    Ps[i] = g_offset * Ps[i];
    Rs[i] = g_offset * Rs[i];
    Vs[i] = g_offset * Vs[i];
  }
  return true;
}

}  // namespace vins
