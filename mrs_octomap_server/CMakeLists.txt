cmake_minimum_required(VERSION 3.5)
project(mrs_octomap_server)

add_definitions(-DCOLOR_OCTOMAP_SERVER)
add_definitions(-DOCTOMAP_NODEBUGOUT)
add_definitions(-O3)

# disables anoying googletest warning
set(CATKIN_ENABLE_TESTING FALSE)

# disables anoying PCL warnings
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")

set(CATKIN_DEPENDENCIES
  cmake_modules
  geometry_msgs
  laser_geometry
  message_generation
  message_runtime
  mrs_lib
  mrs_msgs
  nodelet
  octomap_msgs
  octomap_ros
  mrs_octomap_tools
  pcl_conversions
  pcl_msgs
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  )

set(LIBRARIES
  MrsOctomapServer_Server
  )

find_package(OpenMP REQUIRED)

find_package(octomap REQUIRED)

find_package(PCL REQUIRED COMPONENTS
  common
  )

# include Eigen3
find_package(Eigen3 REQUIRED)
set(Eigen_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
set(Eigen_LIBRARIES ${Eigen_LIBRARIES})

find_package(catkin REQUIRED COMPONENTS
  ${CATKIN_DEPENDENCIES}
  )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# add_compile_options("-fsanitize=address")
# add_link_options("-fsanitize=address")

add_compile_options("${PCL_COMPILE_OPTIONS}")

add_message_files(DIRECTORY msg FILES
  PoseWithSize.msg
)

generate_messages(DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${LIBRARIES}
  CATKIN_DEPENDS ${CATKIN_DEPENDENCIES}
  DEPENDS OCTOMAP
  )

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  )

link_directories(
  ${PCL_LIBRARIES}
)

# Server

add_library(MrsOctomapServer_Server
  src/octomap_server.cpp
  src/conversions.cpp
  )

add_dependencies(MrsOctomapServer_Server
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  )

target_include_directories(MrsOctomapServer_Server PUBLIC
  ${PCL_INCLUDE_DIRS}
  )

target_link_libraries(MrsOctomapServer_Server
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
  )

## --------------------------------------------------------------
## |                           Install                          |
## --------------------------------------------------------------

install(TARGETS ${LIBRARIES}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
  )

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )

install(FILES nodelets.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )
