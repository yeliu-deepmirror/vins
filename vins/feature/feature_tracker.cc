
#include "vins/feature/feature_tracker.h"

namespace vins {
namespace feature {

namespace {

inline cv::Point2f UndistortPointFast(const cv::Point2f& point_xy,
                                      const Eigen::VectorXd& undist_param) {
  if (undist_param.squaredNorm() == 0) {
    return point_xy;
  }
  // https://github.com/opencv/opencv/blob/master/modules/calib3d/src/undistort.dispatch.cpp
  // compensate distortion iteratively (by default opencv process 5 iterations)
  float x = point_xy.x;
  float y = point_xy.y;
  for (int j = 0; j < 5; j++) {
    float x_sqr = x * x;
    float y_sqr = y * y;
    float xy = x * y;

    float r_sqr = x_sqr + y_sqr;
    float icdist =
        1.0 / (1 + ((undist_param(4) * r_sqr + undist_param(1)) * r_sqr + undist_param(0)) * r_sqr);
    if (icdist < 0) {  // test: undistortPoints.regression_14583
      x = point_xy.x;
      y = point_xy.y;
      break;
    }
    float delta_x = 2 * undist_param(2) * xy + undist_param(3) * (r_sqr + 2 * x_sqr);
    float delta_y = undist_param(2) * (r_sqr + 2 * y_sqr) + 2 * undist_param(3) * xy;
    x = (point_xy.x - delta_x) * icdist;
    y = (point_xy.y - delta_y) * icdist;
  }
  return cv::Point2f(x, y);
}

template <typename Type>
inline void ReduceVector(std::vector<Type>& v, const std::vector<uchar>& status) {
  int j = 0;
  for (size_t i = 0; i < v.size(); i++)
    if (status[i]) v[j++] = v[i];
  v.resize(j);
}

template <typename T>
inline std::vector<size_t> ArgSortVector(const std::vector<T>& v) {
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });
  return idx;
}

}  // namespace

int FeatureTracker::n_id = 0;

bool FeatureTracker::InBorder(const cv::Point2f& pt) {
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return 2 <= img_x && img_x < intrinsic_.col_ - 2 && 2 <= img_y && img_y < intrinsic_.row_ - 2;
}

FeatureTracker::FeatureTracker(bool equalize, int max_num_pts, int min_pt_distance)
    : equalize_(equalize), max_num_pts_(max_num_pts), min_pt_distance_(min_pt_distance) {}

void FeatureTracker::GetMaskAndFilterPoints(cv::Mat& mMask) {
  mMask = cv::Mat(intrinsic_.row_, intrinsic_.col_, CV_8UC1, cv::Scalar(255));

  std::vector<cv::Point2f> vForwPts_old = vForwPts;
  std::vector<int> vFeatureIds_old = vFeatureIds;
  std::vector<int> vTrackCnt_old = vTrackCnt;

  vForwPts.clear();
  vFeatureIds.clear();
  vTrackCnt.clear();

  for (auto& it : ArgSortVector(vTrackCnt_old)) {
    cv::Point2f& pt = vForwPts_old[it];
    if (mMask.at<uchar>(pt) == 255) {
      vForwPts.push_back(pt);
      vFeatureIds.push_back(vFeatureIds_old[it]);
      vTrackCnt.push_back(vTrackCnt_old[it]);
      cv::circle(mMask, pt, min_pt_distance_, 0, -1);
    }
  }
}

void FeatureTracker::AddPointsToTrack(std::vector<cv::Point2f>& vNewFeatures) {
  for (auto& p : vNewFeatures) {
    vForwPts.push_back(p);
    vFeatureIds.push_back(-1);
    vTrackCnt.push_back(1);
  }
}

void FeatureTracker::ReadImage(const cv::Mat& _img, double _cur_time, bool bPublish) {
  cv::Mat img;
  cur_time = _cur_time;

  if (equalize_) {
    static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(4, 4));
    clahe->apply(_img, img);
  } else {
    img = _img;
  }

  if (mForwImg.empty()) {
    // for the initial state
    mCurImg = mForwImg = img;
  } else {
    mForwImg = img;
  }

  vForwPts.clear();

  if (vCurPts.size() > 0) {
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(mCurImg, mForwImg, vCurPts, vForwPts, status, err, cv::Size(21, 21),
                             3);

    for (int i = 0; i < int(vForwPts.size()); i++)
      if (status[i] && !InBorder(vForwPts[i])) status[i] = 0;

    ReduceVector(vCurPts, status);
    ReduceVector(vForwPts, status);
    ReduceVector(vFeatureIds, status);
    ReduceVector(vTrackCnt, status);
  }

  // record tracked count
  for (auto& n : vTrackCnt) n++;

  if (bPublish) {
    RejectWithFundamentalMatrix();
    cv::Mat mMask;
    GetMaskAndFilterPoints(mMask);
    // if we want more features for tracking
    int n_max_cnt = max_num_pts_ - static_cast<int>(vForwPts.size());
    if (n_max_cnt > 0) {
      std::vector<cv::Point2f> vNewFeatures;
      cv::goodFeaturesToTrack(mForwImg, vNewFeatures, max_num_pts_ - vForwPts.size(), 0.01,
                              min_pt_distance_, mMask);
      AddPointsToTrack(vNewFeatures);
    }
  }
  mCurImg = mForwImg;
  vCurPts = vForwPts;
  prev_time = cur_time;
  UndistortedPoints();
}

void FeatureTracker::RejectWithFundamentalMatrix() {
  if (vForwPts.size() >= 8) {
    TicToc t_f;
    std::vector<cv::Point2f> un_vCurPts(vCurPts.size()), un_vForwPts(vForwPts.size());
    for (unsigned int i = 0; i < vCurPts.size(); i++) {
      cv::Point2f undist_pt =
          UndistortPointFast(NormalizePoint(vCurPts[i]), intrinsic_.undist_param_);
      un_vCurPts[i] = cv::Point2f(600 * undist_pt.x + intrinsic_.col_ / 2.0,
                                  600 * undist_pt.y + intrinsic_.row_ / 2.0);
      undist_pt = UndistortPointFast(NormalizePoint(vForwPts[i]), intrinsic_.undist_param_);
      un_vForwPts[i] = cv::Point2f(600 * undist_pt.x + intrinsic_.col_ / 2.0,
                                   600 * undist_pt.y + intrinsic_.row_ / 2.0);
    }

    vector<uchar> status;
    cv::findFundamentalMat(un_vCurPts, un_vForwPts, cv::FM_RANSAC, 1.0, 0.99, status);

    ReduceVector(vCurPts, status);
    ReduceVector(vForwPts, status);
    ReduceVector(vFeatureIds, status);
    ReduceVector(vTrackCnt, status);
  }
}

bool FeatureTracker::updateID(unsigned int i) {
  if (i < vFeatureIds.size()) {
    if (vFeatureIds[i] == -1) vFeatureIds[i] = n_id++;
    return true;
  } else {
    return false;
  }
}

void FeatureTracker::UpdateIdMono() {
  for (size_t i = 0; i < vFeatureIds.size(); i++) {
    if (vFeatureIds[i] == -1) vFeatureIds[i] = n_id++;
  }
}

void FeatureTracker::UndistortedPoints() {
  vCurUndistortPts.clear();
  for (unsigned int i = 0; i < vCurPts.size(); i++) {
    cv::Point2f undist_pt =
        UndistortPointFast(NormalizePoint(vCurPts[i]), intrinsic_.undist_param_);
    vCurUndistortPts.emplace_back(std::move(undist_pt));
  }
}

}  // namespace feature
}  // namespace vins
