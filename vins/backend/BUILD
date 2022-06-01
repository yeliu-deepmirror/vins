package(default_visibility = ["//visibility:public"])

cc_library(
    name = "problem",
    srcs = [
        "problem.cc",
    ],
    hdrs = [
        "problem.h",
    ],
    deps = [
        "//vins/backend/common:tic_toc",
        "//vins/backend/vertex:vertex",
        "//vins/backend/edge:edges",
        "@com_github_glog//:glog",
    ],
)

cc_test(
    name = "curve_fitting_test",
    srcs = ["curve_fitting_test.cc"],
    deps = [
        "//vins/backend:problem",
        "@com_google_googletest//:gtest_main",
    ],
)
