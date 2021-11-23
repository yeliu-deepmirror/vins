package(default_visibility = ["//visibility:public"])

cc_library(
    name = "eigen_types",
    srcs = [
        "eigen_types.h"
    ],
    deps = [
        "@eigen",
        "@sophus",
        "@com_github_glog//:glog",
    ],
)

cc_library(
    name = "loss_function",
    srcs = [
        "loss_function.cc",
    ],
    hdrs = [
        "loss_function.h",
    ],
    deps = [
        ":eigen_types",
    ],
)

cc_library(
    name = "utility",
    srcs = [
        "utility.h"
    ],
    deps = [
        ":eigen_types",
    ],
)

cc_library(
    name = "integration_base",
    srcs = [
        "integration_base.cc"
    ],
    hdrs = [
        "integration_base.h",
    ],
    deps = [
        ":utility",
    ],
)

cc_library(
    name = "tic_toc",
    srcs = [
        "tic_toc.h"
    ],
)
