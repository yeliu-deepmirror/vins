#include "vins/initialization/initial_alignment.h"

namespace vins {
namespace backend {

void solveGyroscopeBias(std::map<int64_t, ImageFrame>& all_image_frame, Eigen::Vector3d* Bgs) {
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  Eigen::Vector3d delta_bg;
  std::map<int64_t, ImageFrame>::iterator frame_i;
  std::map<int64_t, ImageFrame>::iterator frame_j;
  for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++) {
    frame_j = next(frame_i);
    Eigen::MatrixXd tmp_A(3, 3);
    tmp_A.setZero();
    Eigen::VectorXd tmp_b(3);
    tmp_b.setZero();
    Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
    tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
    tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
    A += tmp_A.transpose() * tmp_A;
    b += tmp_A.transpose() * tmp_b;
  }
  delta_bg = A.ldlt().solve(b);

  for (int i = 0; i <= feature::WINDOW_SIZE; i++) Bgs[i] += delta_bg;

  for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++) {
    frame_j = next(frame_i);
    frame_j->second.pre_integration->RePropagate(Vector3d::Zero(), Bgs[0]);
  }
}

MatrixXd TangentBasis(Vector3d& g0) {
  Vector3d b, c;
  Vector3d a = g0.normalized();
  Vector3d tmp(0, 0, 1);
  if (a == tmp) tmp << 1, 0, 0;
  b = (tmp - a * (a.transpose() * tmp)).normalized();
  c = a.cross(b);
  MatrixXd bc(3, 2);
  bc.block<3, 1>(0, 0) = b;
  bc.block<3, 1>(0, 1) = c;
  return bc;
}

void RefineGravity(const Eigen::Vector3d& trans_ic, map<int64_t, ImageFrame>& all_image_frame,
                   Vector3d& g, VectorXd& x) {
  Vector3d g0 = g.normalized() * GRAVITY_NORM;
  Vector3d lx, ly;
  // VectorXd x;
  int all_frame_count = all_image_frame.size();
  int n_state = all_frame_count * 3 + 2 + 1;

  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  map<int64_t, ImageFrame>::iterator frame_i;
  map<int64_t, ImageFrame>::iterator frame_j;
  for (int k = 0; k < 4; k++) {
    MatrixXd lxly(3, 2);
    lxly = TangentBasis(g0);
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end();
         frame_i++, i++) {
      frame_j = next(frame_i);

      MatrixXd tmp_A(6, 9);
      tmp_A.setZero();
      VectorXd tmp_b(6);
      tmp_b.setZero();

      double dt = frame_j->second.pre_integration->sum_dt;

      tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
      tmp_A.block<3, 2>(0, 6) =
          frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
      tmp_A.block<3, 1>(0, 8) =
          frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
      tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p +
                                frame_i->second.R.transpose() * frame_j->second.R * trans_ic -
                                trans_ic - frame_i->second.R.transpose() * dt * dt / 2 * g0;

      tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
      tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
      tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
      tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v -
                                frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;

      Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
      cov_inv.setIdentity();

      MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
      VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

      A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
      b.segment<6>(i * 3) += r_b.head<6>();

      A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
      b.tail<3>() += r_b.tail<3>();

      A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
      A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    VectorXd dg = x.segment<2>(n_state - 3);
    g0 = (g0 + lxly * dg).normalized() * GRAVITY_NORM;
  }
  g = g0;
}

bool LinearAlignment(const Eigen::Vector3d& trans_ic, map<int64_t, ImageFrame>& all_image_frame,
                     Vector3d& g, VectorXd& x) {
  int all_frame_count = all_image_frame.size();
  int n_state = all_frame_count * 3 + 3 + 1;

  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  map<int64_t, ImageFrame>::iterator frame_i;
  map<int64_t, ImageFrame>::iterator frame_j;
  int i = 0;
  for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++) {
    frame_j = next(frame_i);

    MatrixXd tmp_A(6, 10);
    tmp_A.setZero();
    VectorXd tmp_b(6);
    tmp_b.setZero();

    double dt = frame_j->second.pre_integration->sum_dt;

    tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
    tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
    tmp_A.block<3, 1>(0, 9) =
        frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
    tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p +
                              frame_i->second.R.transpose() * frame_j->second.R * trans_ic -
                              trans_ic;
    tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
    tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
    tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
    tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;

    Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Identity();
    MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
    VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

    A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
    b.segment<6>(i * 3) += r_b.head<6>();

    A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
    b.tail<4>() += r_b.tail<4>();

    A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
    A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
  }
  A = A * 1000.0;
  b = b * 1000.0;
  x = A.ldlt().solve(b);
  double s = x(n_state - 1) / 100.0;

  g = x.segment<3>(n_state - 4);

  if (fabs(g.norm() - GRAVITY_NORM) > 1.0 || s < 0) {
    return false;
  }

  RefineGravity(trans_ic, all_image_frame, g, x);
  s = (x.tail<1>())(0) / 100.0;
  (x.tail<1>())(0) = s;

  if (s < 0.0)
    return false;
  else
    return true;
}

bool VisualIMUAlignment(const Eigen::Vector3d& trans_ic,
                        std::map<int64_t, ImageFrame>& all_image_frame, Eigen::Vector3d* Bgs,
                        Eigen::Vector3d& g, Eigen::VectorXd& x) {
  solveGyroscopeBias(all_image_frame, Bgs);
  return LinearAlignment(trans_ic, all_image_frame, g, x);
}
}  // namespace backend
}  // namespace vins
