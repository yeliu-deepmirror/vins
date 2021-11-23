package(default_visibility = ["//visibility:public"])

cc_library(
    name = "estimator",
    srcs = [
        "estimator.cc",
    ],
    hdrs = [
        "estimator.h",
        "parameters.h",
    ],
    deps = [
        "//vins/backend:problem",
        "//vins/backend/common:tic_toc",
        "//vins/backend/common:utility",
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
    ],
)

cc_binary(
    name = "run_deepmirror",
    srcs = ["run_deepmirror.cc"],
    deps = [
        ":system",
    ],
)

cc_binary(
    name = "run_record",
    srcs = ["run_record.cc"],
    deps = [
        ":system",
        "//modules/map/data_processor:dso_processor",
        "//modules/common/record:record_reader",
        "@dm_core_cpp//common/base:glog",
        "@dm_core_map//map/session:sensor_utils",
    ],
)
