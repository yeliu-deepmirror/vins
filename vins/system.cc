#include "vins/system.h"

#include <cerrno>
#include <cstdio>
#include <fstream>

using namespace std;
using namespace cv;
using namespace pangolin;

namespace vins {

System::System(const vins::proto::VinsConfig& vins_config)
    : vins_config_(vins_config),
      feature_tracker_(vins_config.equalize(), vins_config.max_num_pts(),
                       vins_config.min_pt_distance()),
      estimator_(vins_config_.verbose(), vins_config.fx()) {
  Eigen::Matrix<double, 6, 1> dist_coeff;
  dist_coeff << vins_config.k1(), vins_config.k2(), vins_config.p1(), vins_config.p2(),
      vins_config.k3(), 0.0;
  // read config from proto
  feature_tracker_.SetIntrinsicParameter(feature::CameraIntrinsic{
      vins_config.image_width(), vins_config.image_height(), vins_config.fx(), vins_config.fy(),
      vins_config.cx(), vins_config.cy(), dist_coeff});

  // set imu extrinsics
  Eigen::Quaterniond qic(vins_config.camera_to_imu().qw(), vins_config.camera_to_imu().qx(),
                         vins_config.camera_to_imu().qy(), vins_config.camera_to_imu().qz());
  estimator_.SetParameter(Sophus::SO3d(qic), Eigen::Vector3d(vins_config.camera_to_imu().x(),
                                                             vins_config.camera_to_imu().y(),
                                                             vins_config.camera_to_imu().z()));
  // set imu intrinsics if has valid input
  if (vins_config.has_acc_noise()) {
    estimator_.imu_intrinsics_.acc_noise = ToEigen(vins_config.acc_noise());
  }
  if (vins_config.has_gyr_noise()) {
    estimator_.imu_intrinsics_.gyr_noise = ToEigen(vins_config.gyr_noise());
  }
  if (vins_config.has_acc_random_walk()) {
    estimator_.imu_intrinsics_.acc_random_walk = ToEigen(vins_config.acc_random_walk());
  }
  if (vins_config.has_gyr_random_walk()) {
    estimator_.imu_intrinsics_.gyr_random_walk = ToEigen(vins_config.gyr_random_walk());
  }
  std::cout << "==> [SYSTEM] system initilize done." << endl;
}

System::~System() {
  pangolin::QuitAll();
  estimator_.ClearState();
}

bool System::PublishImageData(int64_t timestamp, cv::Mat& img, cv::Mat& depth) {
  double stamp_second = static_cast<double>(timestamp) * 1e-9;
  // detect unstable camera stream
  if (stamp_second - current_time_ > 1.0 || stamp_second < current_time_) {
    return false;
  }

  PublishImuData(timestamp, estimator_.acc_0, estimator_.gyr_0);
  if (!depth.empty()) {
    CHECK_EQ(img.cols, depth.cols);
    CHECK_EQ(img.rows, depth.rows);
  }

  feature_tracker_.ReadImage(img, stamp_second, true);
  feature_tracker_.UpdateIdMono();

  std::map<uint64_t, std::vector<std::pair<int, Eigen::Matrix<double, 3, 1>>>> image;
  const auto& un_pts = feature_tracker_.vCurUndistortPts;
  const auto& feature_ids = feature_tracker_.vFeatureIds;
  const auto& pixels = feature_tracker_.vCurPts;
  for (size_t j = 0; j < feature_ids.size(); j++) {
    if (feature_tracker_.vTrackCnt[j] < 2) continue;
    // use -1 for depth to indicate that we have no good initial
    double depth_val = -1.0;
    if (!depth.empty()) {
      depth_val = depth.at<float>(pixels[j].y, pixels[j].x);
    }
    image[feature_ids[j]].emplace_back(0, Eigen::Vector3d(un_pts[j].x, un_pts[j].y, depth_val));
  }

  estimator_.ProcessImage(image, timestamp);

  if (vins_config_.viz()) {
    cv::Mat show_img = img;
    ShowTrack(&show_img);
    cv::imshow("IMAGE", show_img);
    cv::waitKey(1);
  }

  if (estimator_.solver_flag != Estimator::SolverFlag::NON_LINEAR) {
    // clean the poses trajectory
    camera_poses_.clear();
    return false;
  }

  // update frame poses
  for (int i = 0; i < feature::WINDOW_SIZE + 1; i++) {
    Eigen::Matrix3d rot = estimator_.Rs[i] * estimator_.rigid_ic_.so3().matrix();
    Eigen::Vector3d trans =
        estimator_.Rs[i] * estimator_.rigid_ic_.translation() + estimator_.Ps[i];
    camera_poses_[estimator_.Headers[i]] = Sophus::SE3d(Eigen::Quaterniond(rot), trans);
  }
  return true;
}

void System::ShowTrack(cv::Mat* image) {
  CHECK(image != nullptr);
  if (image->channels() == 1) {
    cv::cvtColor(*image, *image, cv::COLOR_GRAY2BGR);
  }
  const auto& pixels = feature_tracker_.vCurPts;
  for (unsigned int j = 0; j < pixels.size(); j++) {
    double len = min(1.0, 1.0 * feature_tracker_.vTrackCnt[j] / feature::WINDOW_SIZE);
    cv::circle(*image, pixels[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
  }
}

bool System::PublishImuData(int64_t timestamp, const Eigen::Vector3d& acc,
                            const Eigen::Vector3d& gyr) {
  double stamp_second = static_cast<double>(timestamp) * 1e-9;
  if (stamp_second <= current_time_) {
    LOG(ERROR) << "imu message in disorder!" << stamp_second << " " << current_time_;
    return false;
  }
  if (current_time_ < 0) current_time_ = stamp_second;
  double dt = stamp_second - current_time_;
  current_time_ = stamp_second;

  estimator_.processIMU(dt, acc, gyr);
  return true;
}

void System::Draw() {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Vins Viewer", 1224, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::CreatePanel("menu_vins").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(200));
  pangolin::Var<bool> menuShowMarPoints("menu_vins.Marginalized Points", true, true);
  pangolin::Var<bool> menuShowCurPoints("menu_vins.Current Points", true, true);
  pangolin::Var<bool> menuShowKeyFrames("menu_vins.KeyFrames", true, true);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
      pangolin::ModelViewLookAt(0, -0.7, -1.8, 0, 0, 0, 0.0, 0.0, 1.0));

  pangolin::View& d_cam =
      pangolin::CreateDisplay()
          .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
          .SetHandler(new pangolin::Handler3D(s_cam));

  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    pangolin::glDrawAxis(3);

    // draw trajectory
    glColor3f(0, 0, 0);
    glLineWidth(2);
    glBegin(GL_LINE_STRIP);
    for (auto& pose_iter : camera_poses_) {
      auto& trans = pose_iter.second.translation();
      glVertex3f(trans(0), trans(1), trans(2));
    }
    glEnd();

    // points
    glPointSize(5);
    glBegin(GL_POINTS);
    if (estimator_.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
      glColor3f(1, 0, 0);
      for (int i = 0; i < feature::WINDOW_SIZE + 1; ++i) {
        Eigen::Vector3d p_wc =
            estimator_.Rs[i] * estimator_.rigid_ic_.translation() + estimator_.Ps[i];
        glVertex3d(p_wc[0], p_wc[1], p_wc[2]);
      }
    }

    if (menuShowMarPoints) {
      glColor3f(0, 0, 0);
      for (auto& iter : estimator_.all_map_points_) {
        auto& pt = iter.second;
        glVertex3d(pt[0], pt[1], pt[2]);
      }
    }

    glEnd();

    pangolin::FinishFrame();
    usleep(50000);  // sleep 5 ms
  }
}

}  // namespace vins
