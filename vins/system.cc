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
      estimator_(vins_config_.verbose()) {
  Eigen::Matrix<double, 6, 1> dist_coeff;
  dist_coeff << vins_config.k1(), vins_config.k2(), vins_config.p1(), vins_config.p2(),
      vins_config.k3(), 0.0;
  // read config from proto
  feature_tracker_.SetIntrinsicParameter(feature::CameraIntrinsic{
      vins_config.image_width(), vins_config.image_height(), vins_config.fx(), vins_config.fy(),
      vins_config.cx(), vins_config.cy(), dist_coeff});

  Eigen::Quaterniond qic(vins_config.camera_to_imu().qw(), vins_config.camera_to_imu().qx(),
                         vins_config.camera_to_imu().qy(), vins_config.camera_to_imu().qz());
  estimator_.SetParameter(Sophus::SO3d(qic), Eigen::Vector3d(vins_config.camera_to_imu().x(),
                                                             vins_config.camera_to_imu().y(),
                                                             vins_config.camera_to_imu().z()));
  std::cout << "==> [SYSTEM] system initilize done." << endl;
}

System::~System() {
  pangolin::QuitAll();
  estimator_.ClearState();
}

bool System::PubImageData(double stamp_second, cv::Mat& img) {
  // detect unstable camera stream
  //   -- that receive the image to late or early image
  if (stamp_second - current_time_ > 1.0 || stamp_second < current_time_) {
    return false;
  }
  PubImuData(stamp_second, latest_acc_, latest_gyr_);

  feature_tracker_.ReadImage(img, stamp_second, true);
  feature_tracker_.UpdateIdMono();

  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 3, 1>>>> image;
  auto& un_pts = feature_tracker_.vCurUndistortPts;
  // auto& vCurPts = feature_tracker_.vCurPts;
  auto& vFeatureIds = feature_tracker_.vFeatureIds;
  // auto& vFeatureVelocity = feature_tracker_.vFeatureVelocity;
  for (size_t j = 0; j < vFeatureIds.size(); j++) {
    if (feature_tracker_.vTrackCnt[j] < 2) continue;
    // int p_id = vFeatureIds[j];
    // double x = un_pts[j].x;
    // double y = un_pts[j].y;
    // Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
    // xyz_uv_velocity << x, y, 1.0, vCurPts[j].x, vCurPts[j].y, vFeatureVelocity[j].x,
    //     vFeatureVelocity[j].y;
    image[vFeatureIds[j]].emplace_back(0, Eigen::Vector3d(un_pts[j].x, un_pts[j].y, 1.0));
  }

  estimator_.ProcessImage(image, stamp_second);

  if (SHOW_TRACK) {
    if (estimator_.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
      Eigen::Vector3d p_wi;
      Eigen::Quaterniond q_wi;
      q_wi = Quaterniond(estimator_.Rs[WINDOW_SIZE]);
      p_wi = estimator_.Ps[WINDOW_SIZE];

      vPath_to_draw.push_back(p_wi);
      keyframe_history.push_back(estimator_.GetCurrentCameraPose());
    }

    cv::Mat show_img;
    cv::cvtColor(img, show_img, cv::COLOR_GRAY2BGR);

    for (unsigned int j = 0; j < feature_tracker_.vCurPts.size(); j++) {
      double len = min(1.0, 1.0 * feature_tracker_.vTrackCnt[j] / WINDOW_SIZE);
      cv::circle(show_img, feature_tracker_.vCurPts[j], 2,
                 cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    cv::imshow("IMAGE", show_img);
    cv::waitKey(1);
  }
  return true;
}

bool System::PubImuData(double stamp_second, const Eigen::Vector3d& acc,
                        const Eigen::Vector3d& gyr) {
  if (stamp_second <= current_time_) {
    cerr << "imu message in disorder!" << stamp_second << " " << current_time_ << endl;
    return false;
  }
  if (current_time_ < 0) {
    current_time_ = stamp_second;
    return true;
  }

  double dt = stamp_second - current_time_;
  current_time_ = stamp_second;

  estimator_.processIMU(dt, acc, gyr);
  latest_acc_ = acc;
  latest_gyr_ = gyr;
  return true;
}

void System::Draw() {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Map Viewer", 1224, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(200));
  pangolin::Var<bool> menuShowMarPoints("menu.Marginalized Points", true, true);
  pangolin::Var<bool> menuShowCurPoints("menu.Current Points", true, true);
  pangolin::Var<bool> menuShowKeyFrames("menu.KeyFrames", true, true);

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
    glColor3f(0, 0, 1);
    pangolin::glDrawAxis(3);

    // draw poses
    glColor3f(0, 0, 0);
    glLineWidth(2);
    glBegin(GL_LINES);
    int nPath_size = vPath_to_draw.size();
    for (int i = 0; i < nPath_size - 1; ++i) {
      glVertex3f(vPath_to_draw[i].x(), vPath_to_draw[i].y(), vPath_to_draw[i].z());
      glVertex3f(vPath_to_draw[i + 1].x(), vPath_to_draw[i + 1].y(), vPath_to_draw[i + 1].z());
    }
    glEnd();

    if (menuShowKeyFrames) {
      nPath_size = keyframe_history.size();
      for (int i = 0; i < nPath_size - 1; i = i + 5) {
        DrawKeyframe(keyframe_history[i]);
      }
    }

    // points
    if (estimator_.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
      glPointSize(5);
      glBegin(GL_POINTS);
      for (int i = 0; i < WINDOW_SIZE + 1; ++i) {
        Vector3d p_wi = estimator_.Ps[i];
        glColor3f(1, 0, 0);
        glVertex3d(p_wi[0], p_wi[1], p_wi[2]);
      }
      glEnd();
    }

    // points
    if (menuShowMarPoints) {
      glPointSize(3);
      glBegin(GL_POINTS);
      glColor3f(0, 0, 0);
      int margalized_size = estimator_.margin_cloud_cloud.size();
      for (int i = 0; i < margalized_size - 1; i++) {
        for (Vec3 p_wi : estimator_.margin_cloud_cloud[i]) {
          glVertex3d(p_wi[0], p_wi[1], p_wi[2]);
        }
      }
      glEnd();

      glPointSize(5);
      glBegin(GL_POINTS);
      glColor3f(1, 0, 0);
      if (margalized_size > 0) {
        for (Vec3 p_wi : estimator_.margin_cloud_cloud[margalized_size - 1]) {
          glVertex3d(p_wi[0], p_wi[1], p_wi[2]);
        }
      }
      glEnd();
    }

    // map points currently seen
    if (menuShowCurPoints) {
      glPointSize(5);
      glBegin(GL_POINTS);
      glColor3f(0, 1, 0);
      for (Vec3 p_wi : estimator_.point_cloud) {
        glVertex3d(p_wi[0], p_wi[1], p_wi[2]);
      }
      glEnd();
    }

    pangolin::FinishFrame();
    usleep(50000);  // sleep 5 ms
  }
}

void System::GetOpenGLCameraMatrix(Eigen::Matrix3d matrix_t, Eigen::Vector3d pVector_t,
                                   pangolin::OpenGlMatrix& M, bool if_inv) {
  Eigen::Matrix3d matrix;
  Eigen::Vector3d pVector;

  // std::cout << matrix_t << std::endl;
  if (if_inv) {
    matrix = matrix_t.transpose();
    pVector = -matrix * pVector_t;
  } else {
    matrix = matrix_t;
    pVector = pVector_t;
  }

  M.m[0] = matrix(0, 0);
  M.m[1] = matrix(1, 0);
  M.m[2] = matrix(2, 0);
  M.m[3] = 0.0;
  M.m[4] = matrix(0, 1);
  M.m[5] = matrix(1, 1);
  M.m[6] = matrix(2, 1);
  M.m[7] = 0.0;
  M.m[8] = matrix(0, 2);
  M.m[9] = matrix(1, 2);
  M.m[10] = matrix(2, 2);
  M.m[11] = 0.0;
  M.m[12] = pVector(0);
  M.m[13] = pVector(1);
  M.m[14] = pVector(2);
  M.m[15] = 1.0;
}

void System::GetOpenGLMatrixCamera(pangolin::OpenGlMatrix& M, Eigen::Matrix<double, 3, 4> Twc) {
  M.m[0] = Twc(0, 0);
  M.m[1] = Twc(1, 0);
  M.m[2] = Twc(2, 0);
  M.m[3] = 0.0;
  M.m[4] = Twc(0, 1);
  M.m[5] = Twc(1, 1);
  M.m[6] = Twc(2, 1);
  M.m[7] = 0.0;
  M.m[8] = Twc(0, 2);
  M.m[9] = Twc(1, 2);
  M.m[10] = Twc(2, 2);
  M.m[11] = 0.0;
  M.m[12] = Twc(0, 3);
  M.m[13] = Twc(1, 3);
  M.m[14] = Twc(2, 3);
  M.m[15] = 1.0;
}

void System::DrawKeyframe(Eigen::Matrix3d matrix_t, Eigen::Vector3d pVector_t) {
  pangolin::OpenGlMatrix currentT;
  GetOpenGLCameraMatrix(matrix_t, pVector_t, currentT, false);

  glPushMatrix();
  glMultMatrixd(currentT.m);
  DrawCamera();
  glPopMatrix();
}

void System::DrawKeyframe(Eigen::Matrix<double, 3, 4> Twc) {
  pangolin::OpenGlMatrix currentT;
  GetOpenGLMatrixCamera(currentT, Twc);

  glPushMatrix();
  glMultMatrixd(currentT.m);
  DrawCamera();
  glPopMatrix();
}

void System::DrawCamera() {
  const float w = mCameraSize;
  const float h = w * 0.75;
  const float z = w * 0.6;
  glLineWidth(mCameraLineWidth);
  glColor3f(0.0f, 1.0f, 0.0f);
  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  glVertex3f(w, h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, h, z);
  glVertex3f(w, h, z);
  glVertex3f(w, -h, z);
  glVertex3f(-w, h, z);
  glVertex3f(-w, -h, z);
  glVertex3f(-w, h, z);
  glVertex3f(w, h, z);
  glVertex3f(-w, -h, z);
  glVertex3f(w, -h, z);
  glEnd();
}

}  // namespace vins
