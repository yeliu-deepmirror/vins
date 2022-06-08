#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <vector>

// estimator
const int WINDOW_SIZE = 10;
const double INIT_DEPTH = 2.0;
const double MIN_PARALLAX = 0.02;  // only threshold for new keyframe #0.02

// from output
const bool ESTIMATE_EXTRINSIC = 0;

// backend
const int NUM_ITERATIONS = 20;
