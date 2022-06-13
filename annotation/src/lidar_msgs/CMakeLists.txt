cmake_minimum_required(VERSION 3.0.2)
project(lidar_msgs)
set (CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED
  COMPONENTS
    geometry_msgs
    sensor_msgs
    std_msgs
    message_generation
)

add_message_files(
  FILES
    PillarTensorTraining.msg
    Params.msg
)


generate_messages(
  DEPENDENCIES
    geometry_msgs
    sensor_msgs
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    sensor_msgs
    message_runtime
    std_msgs
)
