#include <ros/ros.h>
#include <ros/package.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <lidar_msgs/Params.h>
#include <lidar_msgs/PillarTensorTraining.h>

#define PCL_NO_PRECOMPILE

#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>

#include <pcl/common/distances.h>
#include <pcl/common/intersections.h>
#include <pcl/common/io.h>
#include <pcl/common/eigen.h>

#include <pcl/io/pcd_io.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// OpenCV
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <sstream>
#include <iomanip>
// #include <unordered_map>

// namespace
using namespace std;
using namespace pcl;

// define constant
#define _DEG2RAD 0.01745329251

// debug
#define _DEBUG false // false

// predefined function
template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 5)
{
       std::ostringstream out;
       out << std::setprecision(n) << a_value;
       return out.str();
}

std::string to_string_with_sec_nsec(const int sec, const int nsec, const int sec_len=6, const int nsec_len=9)
{
       std::ostringstream ss_sec, ss_nsec;
       ss_sec << std::setw(sec_len) << std::setfill('0') << sec;
       ss_nsec << std::setw(nsec_len) << std::setfill('0') << nsec;

       string str_sec, str_nsec, str_out;
       str_sec = ss_sec.str();
       str_nsec = ss_nsec.str();
       str_out = str_sec + str_nsec;

       return str_out;
}

// own point type
struct OusterPointType
{
       PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding

       float intensity;
       uint16_t reflectivity;
       // float ring;
       // float noise;
       // float range;
       EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
}EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (OusterPointType,           // here we assume a XYZ + "test" (as fields)
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, intensity, intensity)
                                   (uint16_t, reflectivity, reflectivity)
//                                    (float, ring, ring)
//                                    (float, noise, noise)
//                                    (float, range, range)
)

struct PillarPoint
{
       float x;
       float y;
       float z;
       float intensity;
       float xc;
       float yc;
       float zc;
};

// Int32 Par -> Int64
struct IntPairHash
{
	std::size_t operator()(const std::pair<uint32_t, uint32_t> &p) const {
    assert(sizeof(std::size_t)>=8);
    //Shift first integer over to make room for the second integer. The two are
    //then packed side by side.
    return (((uint64_t)p.first)<<32) | ((uint64_t)p.second);
  }
};

// Grid Parameters
float x_min = 0.0f;
float x_max = 46.08f;

float y_min = -11.52f;
float y_max = 11.52f;

float z_min = -3.5f;
float z_max =  0.5f;

float i_min = 0.0f;
float i_max = 8192.f;

float r_min = 0.f;
float r_max = 65535.f;

float x_step = 0.32f; // 32cm 32*144 = 4608 = 46.08m
float y_step = 0.16f; // 16cm 16*144 =  = 15.36m

bool is_x = true;
bool is_y = true;
bool is_z = true;
bool is_i = false;
bool is_r = false;

bool is_img_data_inserted = false;

// Subscribing Messages
static const std::string IMG_CAM = "/camera/image_color/compressed";
static const std::string PC_LIDAR = "/os_cloud_node/points";

// Publisher
ros::Publisher pub_pc_filtered;
ros::Publisher pub_tr_data;

// Variables
int n_file_idx = 0;

// Buffer
cv_bridge::CvImagePtr p_cv_img;

// PointPillars Parameters
int n_max_points_per_pillar = 32;
int n_max_pillars = 20736; // 144*144 = 20736
int n_in_features = 7;

// Priority-based Sampling
bool is_sort_with_z = false;
bool is_sort_with_intensity = false;

// Ascending order (the smallest is at the first)
bool isSortWithZ(PillarPoint a, PillarPoint b)
{
       return a.z < b.z;
}

// Descending order (the largest is at the first)
bool isSortWithIntensity(PillarPoint a, PillarPoint b)
{
       return a.intensity > b.intensity;
}
