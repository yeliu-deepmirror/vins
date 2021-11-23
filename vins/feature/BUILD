package(default_visibility = ["//visibility:public"])

cc_library(
    name = "feature_manager",
    srcs = [
        "feature_manager.cc",
    ],
    hdrs = [
        "feature_manager.h",
    ],
    deps = [
        "@eigen",
    ],
)

cc_library(
    name = "feature_tracker",
    srcs = [
        "feature_tracker.cc",
    ],
    hdrs = [
        "feature_tracker.h",
    ],
    deps = [
        "//vins/backend/common:tic_toc",
        "@opencv",
        "@eigen",
    ],
)
