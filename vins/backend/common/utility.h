#pragma once

#include "vins/backend/common/eigen_types.h"

namespace vins {
namespace backend {

class Utility {
 public:
  template <typename Derived>
  static Eigen::Quaternion<typename Derived::Scalar> deltaQ(
      const Eigen::MatrixBase<Derived>& theta) {
    typedef typename Derived::Scalar Scalar_t;

    Eigen::Quaternion<Scalar_t> dq;
    Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
    half_theta /= static_cast<Scalar_t>(2.0);
    dq.w() = static_cast<Scalar_t>(1.0);
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
  }

  template <typename Derived>
  static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(
      const Eigen::MatrixBase<Derived>& q) {
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1), q(2), typename Derived::Scalar(0), -q(0),
        -q(1), q(0), typename Derived::Scalar(0);
    return ans;
  }

  template <typename Derived>
  static Eigen::Quaternion<typename Derived::Scalar> positify(
      const Eigen::QuaternionBase<Derived>& q) {
    return q;
  }

  template <typename Derived>
  static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(
      const Eigen::QuaternionBase<Derived>& q) {
    Eigen::Quaternion<typename Derived::Scalar> qq = positify(q);
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
    ans.template block<3, 1>(1, 0) = qq.vec(),
                                ans.template block<3, 3>(1, 1) =
                                    qq.w() *
                                        Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() +
                                    skewSymmetric(qq.vec());
    return ans;
  }

  template <typename Derived>
  static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(
      const Eigen::QuaternionBase<Derived>& p) {
    Eigen::Quaternion<typename Derived::Scalar> pp = positify(p);
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
    ans.template block<3, 1>(1, 0) = pp.vec(),
                                ans.template block<3, 3>(1, 1) =
                                    pp.w() *
                                        Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() -
                                    skewSymmetric(pp.vec());
    return ans;
  }

  static Eigen::Vector3d R2ypr(const Eigen::Matrix3d& R) {
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
  }

  template <typename Derived>
  static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(
      const Eigen::MatrixBase<Derived>& ypr) {
    typedef typename Derived::Scalar Scalar_t;

    Scalar_t y = ypr(0) / 180.0 * M_PI;
    Scalar_t p = ypr(1) / 180.0 * M_PI;
    Scalar_t r = ypr(2) / 180.0 * M_PI;

    Eigen::Matrix<Scalar_t, 3, 3> Rz;
    Rz << cos(y), -sin(y), 0, sin(y), cos(y), 0, 0, 0, 1;

    Eigen::Matrix<Scalar_t, 3, 3> Ry;
    Ry << cos(p), 0., sin(p), 0., 1., 0., -sin(p), 0., cos(p);

    Eigen::Matrix<Scalar_t, 3, 3> Rx;
    Rx << 1., 0., 0., 0., cos(r), -sin(r), 0., sin(r), cos(r);

    return Rz * Ry * Rx;
  }

  static Eigen::Matrix3d g2R(const Eigen::Vector3d& g) {
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    return R0;
  }

  inline static Eigen::Matrix3d InverseRightJacobian(const Eigen::Vector3d& phi) {
    const double theta2 = phi.squaredNorm();
    const Eigen::Matrix3d phi_hat = Sophus::SO3d::hat(phi);
    if (theta2 <= 1e-8) {
      return Eigen::Matrix3d::Identity() + 0.5 * phi_hat;
    }
    const double theta = std::sqrt(theta2);
    const Eigen::Vector3d phi_scaled = (1.0 / theta) * phi;
    Eigen::Matrix3d outer_product = phi_scaled * phi_scaled.transpose();
    return Eigen::Matrix3d::Identity() + 0.5 * phi_hat +
           (1 - 0.5 * theta * (1 + std::cos(theta)) / std::sin(theta)) *
               (outer_product - Eigen::Matrix3d::Identity());
  }
};

// marginalize the new frame, which has no visual edge involved
inline void MarginalizeFrameInternal(int idx, int dim, Eigen::MatrixXd* h_prior,
                                     Eigen::VectorXd* b_prior, Eigen::VectorXd* error_prior,
                                     Eigen::MatrixXd* jaco_prior_inv) {
  // skip check for pointer
  CHECK(h_prior != nullptr);
  CHECK(b_prior != nullptr);
  CHECK(error_prior != nullptr);
  CHECK(jaco_prior_inv != nullptr);

  // dim shall be 15:
  // for 3 position, 3 rotation, 3 velocity, 3 bias_acc, 3 bias_gyr
  CHECK_EQ(dim, 15);

  // move the variables to the bottom right
  int reserve_size = h_prior->cols();
  int output_size = reserve_size - dim;
  {
    int tail_size = reserve_size - idx - dim;
    // move the marginalized frame pose to the Hmm bottown right
    // 1. move row i the lowest part
    Eigen::MatrixXd temp_rows = h_prior->block(idx, 0, dim, reserve_size);
    Eigen::MatrixXd temp_botrows = h_prior->block(idx + dim, 0, tail_size, reserve_size);
    h_prior->block(idx, 0, tail_size, reserve_size) = temp_botrows;
    h_prior->block(output_size, 0, dim, reserve_size) = temp_rows;

    // put col i to the rightest part
    Eigen::MatrixXd temp_cols = h_prior->block(0, idx, reserve_size, dim);
    Eigen::MatrixXd temp_rgtcols = h_prior->block(0, idx + dim, reserve_size, tail_size);
    h_prior->block(0, idx, reserve_size, tail_size) = temp_rgtcols;
    h_prior->block(0, output_size, reserve_size, dim) = temp_cols;

    Eigen::VectorXd temp_b = b_prior->segment(idx, dim);
    Eigen::VectorXd temp_btail = b_prior->segment(idx + dim, tail_size);
    b_prior->segment(idx, tail_size) = temp_btail;
    b_prior->segment(output_size, dim) = temp_b;
  }

  static double eps = 1e-8;
  {
    // update h_prior & b_prior
    Eigen::MatrixXd Amm = 0.5 * (h_prior->block(output_size, output_size, dim, dim) +
                                 h_prior->block(output_size, output_size, dim, dim).transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
    Eigen::VectorXd Amm_diag =
        (saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0);
    Eigen::MatrixXd Amm_inv =
        saes.eigenvectors() * Amm_diag.asDiagonal() * saes.eigenvectors().transpose();

    Eigen::VectorXd bmm2 = b_prior->segment(output_size, dim);
    Eigen::MatrixXd Arm = h_prior->block(0, output_size, output_size, dim);
    Eigen::MatrixXd Amr = h_prior->block(output_size, 0, dim, output_size);
    Eigen::MatrixXd Arr = h_prior->block(0, 0, output_size, output_size);
    Eigen::VectorXd brr = b_prior->segment(0, output_size);
    Eigen::MatrixXd tempB = Arm * Amm_inv;
    // the rest pose become the new prior matrix
    // upper we move the rest pose to the upper left, and new pose will be added at lower right.
    // as a result the order of variables of the prior matrix is corresponding to the up coming
    // Hessian.
    *h_prior = Arr - tempB * Amr;
    *b_prior = brr - tempB * bmm2;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(*h_prior);
  auto& saes2_array = saes2.eigenvalues().array();
  Eigen::VectorXd S = Eigen::VectorXd((saes2_array > eps).select(saes2_array, 0));
  Eigen::VectorXd S_inv = Eigen::VectorXd((saes2_array > eps).select(saes2_array.inverse(), 0));

  Eigen::VectorXd S_sqrt = S.cwiseSqrt();
  Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
  *jaco_prior_inv = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
  *error_prior = -(*jaco_prior_inv) * (*b_prior);

  Eigen::MatrixXd J = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
  Eigen::MatrixXd tmp_h = J.transpose() * J;
  *h_prior = MatXX((tmp_h.array().abs() > eps).select(tmp_h.array(), 0));
}

}  // namespace backend
}  // namespace vins
