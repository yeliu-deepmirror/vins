package(default_visibility = ["//visibility:public"])

cc_library(
    name = "initial_alignment",
    srcs = [
        "initial_alignment.cc",
    ],
    hdrs = [
        "initial_alignment.h",
    ],
    deps = [
        "//vins/backend/common:utility",
        "//vins/backend/common:integration_base",
        "//vins/feature:feature_manager",
    ],
)

cc_library(
    name = "initial_sfm",
    srcs = [
        "initial_sfm.cc",
    ],
    hdrs = [
        "initial_sfm.h",
    ],
    deps = [
        "//vins/backend/common:utility",
        "@ceres_solver//:ceres",
        "@opencv",
    ],
)

cc_library(
    name = "solve_5pts",
    srcs = [
        "solve_5pts.cc",
    ],
    hdrs = [
        "solve_5pts.h",
    ],
    deps = [
        "//vins/backend/common:utility",
        "@ceres_solver//:ceres",
        "@opencv",
    ],
)
