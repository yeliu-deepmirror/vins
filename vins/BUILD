package(default_visibility = ["//visibility:public"])

cc_library(
    name = "estimator",
    srcs = [
        "estimator.cc",
        "estimator_optimize.cc",
        "estimator_initialize.cc",
        "estimator_marginalize.cc",
    ],
    hdrs = [
        "estimator.h",
        "parameters.h",
    ],
    deps = [
        "//vins/backend/common:tic_toc",
        "//vins/backend/common:cost_function",
        "//vins/backend/common:loss_function",
        "//vins/feature:feature_manager",
        "//vins/feature:feature_tracker",
        "//vins/initialization:initial_alignment",
        "//vins/initialization:initial_sfm",
        "//vins/initialization:solve_5pts",
    ],
)

cc_library(
    name = "system",
    srcs = [
        "system.cc",
    ],
    hdrs = [
        "system.h",
    ],
    deps = [
        ":estimator",
        "//vins/proto:vins_config_cc_proto",
        "@pangolin",
        "@sophus",
    ],
)
