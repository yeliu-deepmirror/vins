
#include "vins/estimator.h"

namespace vins {

bool Estimator::InitialStructure() {
  TicToc t_sfm;
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
    var = sqrt(var / ((int)all_image_frame.size() - 1));
    if (var < 0.25) {
      return false;
    }
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
    LOG(WARNING) << "Not enough features or parallax; Move device around";
    return false;
  }
  GlobalSFM sfm;
  if (!sfm.construct(frame_count + 1, Q, T, l, relative_R, relative_T, sfm_f, sfm_tracked_points)) {
    LOG(WARNING) << "global SFM failed!";
    marginalization_flag = MARGIN_OLD;
    return false;
  }

  // solve pnp for all frame
  std::map<int64_t, backend::ImageFrame>::iterator frame_it = all_image_frame.begin();
  for (int i = 0; frame_it != all_image_frame.end(); frame_it++) {
    // provide initial guess
    cv::Mat r, rvec, t, D, tmp_r;
    if ((frame_it->first) == Headers[i]) {
      frame_it->second.is_key_frame = true;
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

    frame_it->second.is_key_frame = false;
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
  if (visualInitialAlign())
    return true;
  else {
    LOG(WARNING) << "misalign visual structure with IMU";
    return false;
  }
}

bool Estimator::visualInitialAlign() {
  VectorXd x;
  // solve scale
  if (!VisualIMUAlignment(rigid_ic_.translation(), all_image_frame, Bgs, g, x)) {
    return false;
  }
  // change state
  for (int i = 0; i <= frame_count; i++) {
    Matrix3d Ri = all_image_frame[Headers[i]].R;
    Vector3d Pi = all_image_frame[Headers[i]].T;
    Ps[i] = Pi;
    Rs[i] = Ri;
    all_image_frame[Headers[i]].is_key_frame = true;
  }

  f_manager.ClearDepth();

  // triangulat on cam pose , without extrinsic
  Vector3d TIC_TMP[feature::NUM_OF_CAM];
  for (int i = 0; i < feature::NUM_OF_CAM; i++) TIC_TMP[i].setZero();
  f_manager.triangulate(Ps, rigid_ic_.translation(), rigid_ic_.so3().matrix());

  double s = (x.tail<1>())(0);
  for (int i = 0; i <= feature::WINDOW_SIZE; i++) {
    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
  }
  for (int i = frame_count; i >= 0; i--) {
    Ps[i] =
        s * Ps[i] - Rs[i] * rigid_ic_.translation() - (s * Ps[0] - Rs[0] * rigid_ic_.translation());
  }
  int kv = -1;
  for (std::map<int64_t, backend::ImageFrame>::iterator frame_i = all_image_frame.begin();
       frame_i != all_image_frame.end(); frame_i++) {
    if (frame_i->second.is_key_frame) {
      kv++;
      Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
    }
  }
  for (auto& it_per_id : f_manager.feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < feature::WINDOW_SIZE - 2)) continue;
    it_per_id.estimated_depth *= s;
  }

  Matrix3d R0 = backend::Utility::g2R(g);
  double yaw = backend::Utility::R2ypr(R0 * Rs[0]).x();
  R0 = backend::Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
  g = R0 * g;
  // Matrix3d rot_diff = R0 * Rs[0].transpose();
  Matrix3d rot_diff = R0;
  for (int i = 0; i <= frame_count; i++) {
    Ps[i] = rot_diff * Ps[i];
    Rs[i] = rot_diff * Rs[i];
    Vs[i] = rot_diff * Vs[i];
  }

  return true;
}

bool Estimator::relativePose(Matrix3d& relative_R, Vector3d& relative_T, int& l) {
  // find previous frame which contians enough correspondance and parallex with newest frame
  for (int i = 0; i < feature::WINDOW_SIZE; i++) {
    vector<pair<Vector3d, Vector3d>> corres;
    corres = f_manager.GetCorresponding(i, feature::WINDOW_SIZE);
    if (corres.size() > 20) {
      double sum_parallax = 0;
      double average_parallax;
      for (int j = 0; j < int(corres.size()); j++) {
        Vector2d pts_0(corres[j].first(0), corres[j].first(1));
        Vector2d pts_1(corres[j].second(0), corres[j].second(1));
        double parallax = (pts_0 - pts_1).norm();
        sum_parallax = sum_parallax + parallax;
      }
      average_parallax = 1.0 * sum_parallax / int(corres.size());
      if (average_parallax * 460 > 30 &&
          m_estimator.solveRelativeRT(corres, relative_R, relative_T)) {
        l = i;
        return true;
      }
    }
  }
  return false;
}

}  // namespace vins
