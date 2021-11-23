package(default_visibility = ["//visibility:public"])

cc_library(
    name = "edge",
    srcs = [
        "edge.cc",
    ],
    hdrs = [
        "edge.h",
    ],
    deps = [
        "//vins/backend/common:loss_function",
        "//vins/backend/vertex:vertex",
    ],
)

cc_library(
    name = "edges",
    srcs = [
        "edge_imu.cc",
        "edge_mapfusion.cc",
        "edge_prior.cc",
        "edge_reprojection.cc",
    ],
    hdrs = [
        "edge_imu.h",
        "edge_mapfusion.h",
        "edge_prior.h",
        "edge_reprojection.h",
    ],
    deps = [
        ":edge",
        "//vins/backend/common:integration_base",
    ],
)
