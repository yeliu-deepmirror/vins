
#include "vins/estimator.h"

#include <fstream>
#include <ostream>

namespace vins {

Estimator::Estimator(bool verbose) : verbose_(verbose), gravity_(0, 0, 9.81), f_manager{Rs} {
  LOG(INFO) << "[VINS ESTIMATOR] initialize done.";
  ClearState(true);
  depth_information_(0, 0) = 100.0;
  project_sqrt_info_ = 600.0 / 1.5 * Matrix2d::Identity();
}

void Estimator::SetParameter(const Sophus::SO3d& ric, const Eigen::Vector3d& tic) {
  Quaterniond q_ic = ric.unit_quaternion();
  vPic[0] << tic(0), tic(1), tic(2), q_ic.x(), q_ic.y(), q_ic.z(), q_ic.w();
  rigid_ic_ = Sophus::SE3d(ric, tic);

  project_sqrt_info_ = 600.0 / 1.5 * Matrix2d::Identity();
}

void Estimator::ClearState(bool bInit) {
  for (int i = 0; i < feature::WINDOW_SIZE + 1; i++) {
    Rs[i].setIdentity();
    Ps[i].setZero();
    Vs[i].setZero();
    Bas[i].setZero();
    Bgs[i].setZero();
    dt_buf[i].clear();
    linear_acceleration_buf[i].clear();
    angular_velocity_buf[i].clear();
    if (!bInit) {
      if (pre_integrations[i] != nullptr) {
        delete pre_integrations[i];
      }
    }
    pre_integrations[i] = nullptr;
  }

  if (!bInit) {
    for (auto& it : all_image_frame) {
      if (it.second.pre_integration != nullptr) {
        delete it.second.pre_integration;
        it.second.pre_integration = nullptr;
      }
    }
  }

  solver_flag = INITIAL;
  first_imu = false;
  frame_count = 0;
  solver_flag = INITIAL;
  initial_timestamp = 0;
  all_image_frame.clear();

  if (!bInit) {
    if (tmp_pre_integration != nullptr) delete tmp_pre_integration;
  }
  tmp_pre_integration = nullptr;

  f_manager.ClearState();

  failure_occur = 0;
}

void Estimator::processIMU(double dt, const Vector3d& linear_acceleration,
                           const Vector3d& angular_velocity) {
  if (!first_imu) {
    first_imu = true;
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
  }

  if (!pre_integrations[frame_count]) {
    pre_integrations[frame_count] =
        new backend::IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count], gravity_};
  }

  if (frame_count != 0) {
    pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
    tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

    dt_buf[frame_count].push_back(dt);
    linear_acceleration_buf[frame_count].push_back(linear_acceleration);
    angular_velocity_buf[frame_count].push_back(angular_velocity);

    int j = frame_count;
    Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - gravity_;
    Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
    Rs[j] *= backend::Utility::deltaQ(un_gyr * dt).toRotationMatrix();
    Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - gravity_;
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
    Vs[j] += dt * un_acc;
  }
  acc_0 = linear_acceleration;
  gyr_0 = angular_velocity;
}

void Estimator::ProcessImage(
    const std::map<int, std::vector<std::pair<int, Eigen::Vector3d>>>& image, int64_t header,
    const std::optional<backend::ImuState>& imu_state) {
  if (f_manager.AddFeatureCheckParallax(frame_count, image, 0.0))
    marginalization_flag = MARGIN_OLD;
  else
    marginalization_flag = MARGIN_SECOND_NEW;

  Headers[frame_count] = header;
  if (solver_flag == NON_LINEAR) {
    const auto& last_imu_state = all_image_frame[Headers[frame_count - 1]].imu_state_;
    // set initial pose by odometry pose
    if (imu_state.has_value() && last_imu_state.has_value()) {
      Sophus::SE3d this_to_last = last_imu_state->pose.inverse() * imu_state->pose;
      // set the init guess of the new frame
      Ps[frame_count] = Ps[frame_count - 1] + Rs[frame_count - 1] * this_to_last.translation();
      Rs[frame_count] = Rs[frame_count - 1] * this_to_last.so3().matrix();
    }
  }

  backend::ImageFrame imageframe(image, header);
  imageframe.pre_integration = tmp_pre_integration;
  imageframe.imu_state_ = imu_state;
  all_image_frame.emplace(header, std::move(imageframe));

  tmp_pre_integration =
      new backend::IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count], gravity_};

  if (solver_flag == INITIAL) {
    if (frame_count == feature::WINDOW_SIZE) {
      bool result = false;
      if (header - initial_timestamp > 1e8) {
        result = InitialStructure();
        initial_timestamp = header;
      }
      if (result) {
        solver_flag = NON_LINEAR;
        SolveOdometry();
        slideWindow();
        f_manager.removeFailures();
        LOG(INFO) << "[VINS] Initialization finish!";
        last_R = Rs[feature::WINDOW_SIZE];
        last_P = Ps[feature::WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
      } else {
        slideWindow();
      }
    } else {
      frame_count++;
    }
  } else {
    SolveOdometry();
    if (failureDetection()) {
      failure_occur = 1;
      ClearState();
      return;
    }

    slideWindow();
    f_manager.removeFailures();

    last_R = Rs[feature::WINDOW_SIZE];
    last_P = Ps[feature::WINDOW_SIZE];
    last_R0 = Rs[0];
    last_P0 = Ps[0];
  }
}

void Estimator::SolveOdometry() {
  if (frame_count < feature::WINDOW_SIZE) return;
  if (solver_flag == NON_LINEAR) {
    f_manager.triangulate(Ps, rigid_ic_.translation(), rigid_ic_.so3().matrix());
    BackendOptimization();
  }
}

bool Estimator::failureDetection() {
  if (f_manager.last_track_num < 2) {
    return true;
  }
  if (Bas[feature::WINDOW_SIZE].norm() > 2.5) {
    return true;
  }
  if (Bgs[feature::WINDOW_SIZE].norm() > 1.0) {
    return true;
  }
  Vector3d tmp_P = Ps[feature::WINDOW_SIZE];
  if ((tmp_P - last_P).norm() > 5) {
    return true;
  }
  if (abs(tmp_P.z() - last_P.z()) > 1) {
    return true;
  }
  Matrix3d tmp_R = Rs[feature::WINDOW_SIZE];
  Matrix3d delta_R = tmp_R.transpose() * last_R;
  Quaterniond delta_Q(delta_R);
  double delta_angle;
  delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
  if (delta_angle > 50) {
    return true;
  }
  return false;
}

void Estimator::slideWindow() {
  if (marginalization_flag == MARGIN_OLD) {
    int64_t t_0 = Headers[0];
    Eigen::Matrix3d back_R0 = Rs[0];
    Eigen::Vector3d back_P0 = Ps[0];
    if (frame_count == feature::WINDOW_SIZE) {
      for (int i = 0; i < feature::WINDOW_SIZE; i++) {
        Rs[i].swap(Rs[i + 1]);
        std::swap(pre_integrations[i], pre_integrations[i + 1]);
        dt_buf[i].swap(dt_buf[i + 1]);
        linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
        angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);
        Headers[i] = Headers[i + 1];
        Ps[i].swap(Ps[i + 1]);
        Vs[i].swap(Vs[i + 1]);
        Bas[i].swap(Bas[i + 1]);
        Bgs[i].swap(Bgs[i + 1]);
      }
      Headers[feature::WINDOW_SIZE] = Headers[feature::WINDOW_SIZE - 1];
      Ps[feature::WINDOW_SIZE] = Ps[feature::WINDOW_SIZE - 1];
      Vs[feature::WINDOW_SIZE] = Vs[feature::WINDOW_SIZE - 1];
      Rs[feature::WINDOW_SIZE] = Rs[feature::WINDOW_SIZE - 1];
      Bas[feature::WINDOW_SIZE] = Bas[feature::WINDOW_SIZE - 1];
      Bgs[feature::WINDOW_SIZE] = Bgs[feature::WINDOW_SIZE - 1];

      delete pre_integrations[feature::WINDOW_SIZE];
      pre_integrations[feature::WINDOW_SIZE] = new backend::IntegrationBase{
          acc_0, gyr_0, Bas[feature::WINDOW_SIZE], Bgs[feature::WINDOW_SIZE], gravity_};

      dt_buf[feature::WINDOW_SIZE].clear();
      linear_acceleration_buf[feature::WINDOW_SIZE].clear();
      angular_velocity_buf[feature::WINDOW_SIZE].clear();

      if (true || solver_flag == INITIAL) {
        std::map<int64_t, backend::ImageFrame>::iterator it_0 = all_image_frame.find(t_0);
        CHECK(it_0 != all_image_frame.end());
        delete it_0->second.pre_integration;
        it_0->second.pre_integration = nullptr;

        for (std::map<int64_t, backend::ImageFrame>::iterator it = all_image_frame.begin();
             it != it_0; ++it) {
          if (it->second.pre_integration) delete it->second.pre_integration;
          it->second.pre_integration = NULL;
        }

        all_image_frame.erase(all_image_frame.begin(), it_0);
        all_image_frame.erase(t_0);
      }
      slideWindowOld(back_R0, back_P0);
    }
  } else {
    if (frame_count == feature::WINDOW_SIZE) {
      for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++) {
        double tmp_dt = dt_buf[frame_count][i];
        Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
        Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

        pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration,
                                                     tmp_angular_velocity);

        dt_buf[frame_count - 1].push_back(tmp_dt);
        linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
        angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
      }

      Headers[frame_count - 1] = Headers[frame_count];
      Ps[frame_count - 1] = Ps[frame_count];
      Vs[frame_count - 1] = Vs[frame_count];
      Rs[frame_count - 1] = Rs[frame_count];
      Bas[frame_count - 1] = Bas[frame_count];
      Bgs[frame_count - 1] = Bgs[frame_count];

      delete pre_integrations[feature::WINDOW_SIZE];
      pre_integrations[feature::WINDOW_SIZE] = new backend::IntegrationBase{
          acc_0, gyr_0, Bas[feature::WINDOW_SIZE], Bgs[feature::WINDOW_SIZE], gravity_};

      dt_buf[feature::WINDOW_SIZE].clear();
      linear_acceleration_buf[feature::WINDOW_SIZE].clear();
      angular_velocity_buf[feature::WINDOW_SIZE].clear();

      slideWindowNew();
    }
  }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew() { f_manager.removeFront(frame_count); }

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld(const Eigen::Matrix3d& back_R0, const Eigen::Vector3d& back_P0) {
  if (solver_flag == NON_LINEAR) {
    Eigen::Matrix3d R0 = back_R0 * rigid_ic_.so3().matrix();
    Eigen::Matrix3d R1 = Rs[0] * rigid_ic_.so3().matrix();
    Eigen::Vector3d P0 = back_P0 + back_R0 * rigid_ic_.translation();
    Eigen::Vector3d P1 = Ps[0] + Rs[0] * rigid_ic_.translation();
    f_manager.RemoveBackShiftDepth(R0, P0, R1, P1);
  } else {
    f_manager.RemoveBack();
  }
}

}  // namespace vins
