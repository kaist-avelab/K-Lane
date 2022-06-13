cmake_minimum_required(VERSION 3.0.2)
project(pc_pre_processor)
set (CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  pcl_conversions
  pcl_ros
  roscpp
  image_transport
  std_msgs
  sensor_msgs
  visualization_msgs
  nav_msgs
  cv_bridge
  lidar_msgs
)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/pc_pre_processor.cpp)
add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES} ${OpenCV_LIBRARIES}
)
