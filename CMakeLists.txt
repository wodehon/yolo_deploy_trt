cmake_minimum_required(VERSION 3.0.2)
project(yolo_deploy_trt)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 14) # 或者 17，根据需要调整
add_compile_options(-std=c++14) # 确保和 CMAKE_CXX_STANDARD 一致
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14")
set(CMAKE_BUILD_TYPE Debug)

# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# For NVIDIA Jetson
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_CUDA_ARCHITECTURES "72")

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
# TODO(Call for PR): make TRT path configurable from command line
include_directories(/usr/include/aarch64-linux-gnu/)
link_directories(/usr/lib/aarch64-linux-gnu/)

# cv_bridge for opencv4.5.5
set(cv_bridge_DIR /usr/local/share/cv_bridge/cmake)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  image_transport
  sensor_msgs
  message_generation
  geometry_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

# OpenCV
set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(CUDA REQUIRED)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

file(GLOB_RECURSE UTILS ${PROJECT_SOURCE_DIR}/include/utils/*.cpp ${PROJECT_SOURCE_DIR}/include/utils/*.cu)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/include/plugins/*.cu)

# # TensorRT
# set(TENSORRT_LIB /usr/lib/aarch64-linux-gnu)
# include_directories(/usr/include/aarch64-linux-gnu)

include_directories(
  include
  include/yolov5_trt
  include/utils
  include/plugins
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins ${catkin_LIBRARIES} nvinfer cudart)

# msg
add_message_files(
  FILES
  Detection2D.msg
  Detections2D.msg
)
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)


catkin_package(
  # INCLUDE_DIRS include include/utils include/plugins include/yolov5_trt
  # LIBRARIES yolo_deploy_trt
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs cv_bridge image_transport message_runtime
  # DEPENDS system_lib
)

add_executable(yolov5 src/yolov5_trt_node.cpp src/yolov5_trt_ros.cpp ${UTILS})
target_link_libraries(yolov5
  ${catkin_LIBRARIES}
  nvinfer
  cudart
  myplugins
  ${OpenCV_LIBS}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

