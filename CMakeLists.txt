cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(pointnet-gpd-ros)

find_package(
  catkin REQUIRED
  COMPONENTS
  roscpp
  sensor_msgs
  tf
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  grasp_detect # grasp_detect
)


catkin_package(
  # INCLUDE_DIRS include
#  LIBRARIES grasp_detect
 CATKIN_DEPENDS roscpp sensor_msgs
#  DEPENDS system_lib
)

## Specify additional locations of header files
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
