cmake_minimum_required(VERSION 2.8.3)
project(fused_localization)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  duckietown_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

catkin_package()