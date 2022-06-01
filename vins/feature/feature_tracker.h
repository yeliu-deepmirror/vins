#ifndef VINS_INCLUDE_FEATURE_TRACKER_H_
#define VINS_INCLUDE_FEATURE_TRACKER_H_

#include <iostream>
#include <numeric>
#include <queue>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "vins/backend/common/tic_toc.h"

using namespace std;
using namespace Eigen;

namespace vins {
namespace feature {

struct CameraIntrinsic {
  int col_;
  int row_;
  float fx_;
  float fy_;
  float cx_;
  float cy_;
  Eigen::VectorXd undist_param_;
};

class FeatureTracker {
 public:
  FeatureTracker(bool equalize, int max_num_pts, int min_pt_distance);

  void ReadImage(const cv::Mat& _img, double _cur_time, bool bPublish);
  void UndistortedPoints();

  void GetMaskAndFilterPoints(cv::Mat& mMask);
  void AddPointsToTrack(std::vector<cv::Point2f>& vNewFeatures);
  void RejectWithFundamentalMatrix();

  // update the vFeatureIds of the features (this complicate function allow correct update for
  // multiple cameras)
  bool updateID(unsigned int i);

  // update Ids as linear process, only for monocular camera case
  void UpdateIdMono();

  void SetIntrinsicParameter(const CameraIntrinsic& intrinsic) { intrinsic_ = intrinsic; }

 public:
  bool InBorder(const cv::Point2f& pt);

  double cur_time;
  double prev_time;
  cv::Mat mCurImg, mForwImg;
  std::vector<cv::Point2f> vCurPts, vForwPts;
  std::vector<int> vFeatureIds;
  std::vector<int> vTrackCnt;

  // saved for publisher
  std::vector<cv::Point2f> vCurUndistortPts;

  cv::Point2f NormalizePoint(const cv::Point2f& pt_pixel) {
    return cv::Point2f((pt_pixel.x - intrinsic_.cx_) / intrinsic_.fx_,
                       (pt_pixel.y - intrinsic_.cy_) / intrinsic_.fy_);
  }

  CameraIntrinsic intrinsic_;

  static int n_id;

  // options
  bool equalize_ = true;
  int max_num_pts_ = 500;
  int min_pt_distance_ = 10;
};

}  // namespace feature
}  // namespace vins

#endif  // VINS_INCLUDE_FEATURE_TRACKER_H_
