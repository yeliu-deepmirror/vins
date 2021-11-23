// Copyright 2021 DeepMirror Inc. All rights reserved.

#include "common/file/file.h"
#include "common/file/recordio.h"

#include "map/session/sensor_utils.h"
#include "modules/common/record/record_reader.h"
#include "modules/map/data_processor/dso_processor.h"
#include "proto/drivers/lidar.pb.h"
#include "vins/system.h"

DEFINE_string(session_name, "", "session_name");

DEFINE_string(camera_channel, "/Camera/camera_front/frame", "lidar channel");

DEFINE_string(imu_channel, "/IMU/clapb7_imu/imu", "imu channel");

DEFINE_double(resize_ratio, 0.5, "resize ratio");

/*
SESSION_NAME=20211104T142127+0800_xavier_xavier002_nanshan

bazel run -c opt //vins:run_record -- \
-map_storage_input_directories=/DeepMirror/Server/gz01/raw/ \
-session_name=${SESSION_NAME} \
-camera_channel=/Camera/camera_front/image \
-camera_regexes=camera_front

*/

std::string GetSensorPreFix(const std::string& channel_name) {
  std::string result;
  std::vector<std::string> strings = dm::utils::StrSplit(channel_name, '/');
  for (size_t i = 0; i < strings.size() - 1; i++) {
    if (i > 1) result += ".";
    result += strings[i];
  }
  return result;
}

void TryInverseImu(dm::map::ImuData* imu_data) {
  static bool do_inverse = imu_data->linear_acceleration_data().z() < 0;
  if (do_inverse) {
    *imu_data = dm::map::ImuData(imu_data->timestamp(), -imu_data->linear_acceleration_data(),
                                 imu_data->angular_velocity_data());
  }
  return;
}

int main(int argc, char** argv) {
  DM_InitGoogleLogging(argc, argv);

  auto session = std::make_shared<dm::map::Session>(FLAGS_session_name);

  dm::api::proto::vision::CameraIntrinsics intrinsic;
  dm::internal::proto::sensor::SensorExtrinsics extrinsic_cam;
  dm::internal::proto::sensor::SensorExtrinsics extrinsic_imu;
  std::string camera_pre_fix = GetSensorPreFix(FLAGS_camera_channel);
  CHECK_OK(session->ReadProto(camera_pre_fix + ".intrinsics", &intrinsic));
  CHECK_OK(session->ReadProto(camera_pre_fix + ".extrinsics", &extrinsic_cam));
  CHECK_OK(session->ReadProto(GetSensorPreFix(FLAGS_imu_channel) + ".extrinsics", &extrinsic_imu));

  Sophus::SE3d camera_to_base = mobili::map::ReadExtrinsics(extrinsic_cam).inverse();
  Sophus::SE3d base_to_imu = mobili::map::ReadExtrinsics(extrinsic_imu);
  Sophus::SE3d camera_to_imu = base_to_imu * camera_to_base;

  vins::proto::VinsConfig vins_config;
  vins_config.set_image_width(1920 * FLAGS_resize_ratio);
  vins_config.set_image_height(1080 * FLAGS_resize_ratio);
  vins_config.set_fx(intrinsic.fx() * FLAGS_resize_ratio);
  vins_config.set_fy(intrinsic.fy() * FLAGS_resize_ratio);
  vins_config.set_cx(intrinsic.cx() * FLAGS_resize_ratio);
  vins_config.set_cy(intrinsic.cy() * FLAGS_resize_ratio);
  vins_config.mutable_camera_to_imu()->set_x(camera_to_imu.translation()(0));
  vins_config.mutable_camera_to_imu()->set_y(camera_to_imu.translation()(1));
  vins_config.mutable_camera_to_imu()->set_z(camera_to_imu.translation()(2));
  auto quad = camera_to_imu.so3().unit_quaternion();
  vins_config.mutable_camera_to_imu()->set_qw(quad.w());
  vins_config.mutable_camera_to_imu()->set_qx(quad.x());
  vins_config.mutable_camera_to_imu()->set_qy(quad.y());
  vins_config.mutable_camera_to_imu()->set_qz(quad.z());

  LOG(INFO) << "---  VINS start  ---";
  std::shared_ptr<vins::System> vins_system = std::make_shared<vins::System>(vins_config);

  std::thread thd_BackEnd(&vins::System::ProcessBackEnd, vins_system);
  std::thread thd_Draw(&vins::System::Draw, vins_system);

  // read all message
  mobili::common::RecordReader reader(session);
  apollo::cyber::record::RecordMessage message;
  while (reader.ReadMessage(&message)) {
    if (message.channel_name == FLAGS_camera_channel) {
      mobili::proto::drivers::ImageData raw_proto;
      raw_proto.ParseFromString(message.content);

      cv::Mat image_raw;
      mobili::utils::GetCVImage(raw_proto.image_data().image(), &image_raw);
      cv::cvtColor(image_raw, image_raw, CV_BGR2GRAY);
      cv::resize(
          image_raw, image_raw,
          cv::Size(image_raw.cols * FLAGS_resize_ratio, image_raw.rows * FLAGS_resize_ratio));

      double timestamp_sec = 1.0e-9 * static_cast<double>(raw_proto.timestamp_nsec());
      vins_system->PubImageData(timestamp_sec, image_raw);

    } else if (message.channel_name == FLAGS_imu_channel) {
      mobili::proto::drivers::ImuData input_proto;
      input_proto.ParseFromString(message.content);

      dm::map::ImuData imu_data(input_proto.timestamp_nsec(), input_proto.imu_data());
      TryInverseImu(&imu_data);

      double timestamp_sec = 1.0e-9 * static_cast<double>(input_proto.timestamp_nsec());
      vins_system->PubImuData(timestamp_sec, imu_data.angular_velocity_data(),
                              imu_data.linear_acceleration_data());
    }
  }

  LOG(INFO) << "Process Done.";
  return 0;
}
