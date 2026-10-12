cmake_minimum_required(VERSION 3.5)
project(mybot_velocity_smoother_trace)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_util REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(velocity_smoother_trace src/main.cpp src/velocity_smoother.cpp)
target_include_directories(velocity_smoother_trace BEFORE PRIVATE include)
ament_target_dependencies(velocity_smoother_trace
  rclcpp rclcpp_components geometry_msgs nav2_util)
install(TARGETS velocity_smoother_trace RUNTIME DESTINATION lib/${PROJECT_NAME})
ament_package()
