
#include "vins/system.h"

#include <fcntl.h>
#include <sys/stat.h>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"

using namespace std;
using namespace cv;
using namespace Eigen;

const int nDelayTimes = 1;
string sConfig_path = "/mobili/vins/config/deepmirror.proto.txt";

std::string imu_file = "/DeepMirror/DeepMirror/MapData/20201126T102742+0800_basement/imu.txt";
std::string image_folder =
    "/DeepMirror/DeepMirror/MapData/20201126T102742+0800_basement/images/camera_front/";
std::string image_file =
    "/DeepMirror/DeepMirror/MapData/20201126T102742+0800_basement/images/camera_front_images.txt";

vins::System* pSystem;

bool ReadProtoFromTextFile(const std::string& file_name, google::protobuf::Message* proto) {
  int fd = open(file_name.c_str(), O_RDONLY);
  if (fd == -1) return false;
  google::protobuf::io::FileInputStream input(fd);
  input.SetCloseOnDelete(true);
  if (!google::protobuf::TextFormat::Parse(&input, proto)) return false;
  return true;
}

void PubImuData() {
  std::cout << "==> [IMU] PubImuData start sImu_data_filea: " << imu_file << std::endl;
  std::ifstream fs_imu;
  fs_imu.open(imu_file.c_str());
  if (!fs_imu.is_open()) {
    std::cerr << "Failed to open imu data file! " << imu_file << std::endl;
    return;
  }

  std::string fsimu_line;
  double dStampNSec = 0.0;
  Vector3d vAcc;
  Vector3d vGyr;
  while (std::getline(fs_imu, fsimu_line) && !fsimu_line.empty()) {
    std::istringstream ssImuData(fsimu_line);
    ssImuData >> dStampNSec >> vGyr.x() >> vGyr.y() >> vGyr.z() >> vAcc.x() >> vAcc.y() >> vAcc.z();
    // cout << "Imu t: " << fixed << dStampNSec << " gyr: " << vGyr.transpose() << " acc: " <<
    // vAcc.transpose() << endl;
    pSystem->PubImuData(dStampNSec / 1e9, vGyr, vAcc);
    usleep(2000 * nDelayTimes);
  }
  fs_imu.close();
}

void PubImageData() {
  std::cout << "==> [IMAGE] PubImageData start sImage_file: " << image_file << std::endl;
  std::ifstream fsImage;
  fsImage.open(image_file.c_str());
  if (!fsImage.is_open()) {
    cerr << "Failed to open image paths file! " << image_file << endl;
    return;
  }

  std::string sImage_line;
  int64_t dStampNSec;
  string sImgFileName;

  // cv::namedWindow("SOURCE IMAGE", CV_WINDOW_AUTOSIZE);
  while (std::getline(fsImage, sImage_line) && !sImage_line.empty()) {
    std::istringstream ssImuData(sImage_line);
    ssImuData >> dStampNSec >> sImgFileName;
    string imagePath = image_folder + std::to_string(dStampNSec) + ".jpg";
    // cout << "Image t : " << fixed << dStampNSec << " Name: " << imagePath << endl;

    Mat img = imread(imagePath.c_str(), 0);
    if (img.empty()) {
      cerr << "image is empty! path: " << imagePath << endl;
      return;
    }
    cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2));
    pSystem->PubImageData(static_cast<double>(dStampNSec) / 1e9, img);
    // cv::imshow("SOURCE IMAGE", img);
    // cv::waitKey(0);
    usleep(20000 * nDelayTimes);
  }
  fsImage.close();
}

int main(int argc, char** argv) {
  cout << "  ---  VINS start  ---" << std::endl;
  std::cout << "==> [SYSTEM] set the configuration from file : " << sConfig_path << endl;
  // read config from proto
  vins::proto::VinsConfig vins_config;
  ReadProtoFromTextFile(sConfig_path, &vins_config);

  pSystem = new vins::System(vins_config);

  // sleep(5);
  std::thread thd_BackEnd(&vins::System::ProcessBackEnd, pSystem);

  std::thread thd_PubImuData(PubImuData);

  std::thread thd_PubImageData(PubImageData);

  std::thread thd_Draw(&vins::System::Draw, pSystem);

  thd_PubImuData.join();
  thd_PubImageData.join();

  thd_BackEnd.join();
  thd_Draw.join();

  cout << "main end... see you ..." << endl;
  return 0;
}
