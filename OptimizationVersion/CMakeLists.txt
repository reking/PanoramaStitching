cmake_minimum_required(VERSION 2.8)
project(Mosaic)
find_package( OpenCV REQUIRED)
find_package( ceres REQUIRED)
#find_package( glog REQUIRED)
#FIND_PACKAGE(Eigen REQUIRED)
include_directories(/usr/local/include/eigen3/Eigen)

#PKG_CHECK_MODULES( EIGEN3 REQUIRED eigen3 )
#include_directories( ${EIGEN3_INCLUDE_DIRS} )
#find_package( eigen3 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
#INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIRS})
#LINK_DIRECTORIES(${EIGEN3_LIBRARY_DIRS})
#SET(CMAKE_CXX_COMPILER "/usr/local/bin/g++-4.9")
#SET (CMAKE_CXX_FLAGS    "-stdlib=libstdc++")
add_executable(esti esti.cc)
target_link_libraries(esti ${OpenCV_LIBS} ceres)
