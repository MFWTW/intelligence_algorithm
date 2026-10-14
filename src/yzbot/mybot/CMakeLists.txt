cmake_minimum_required(VERSION 3.22)
project(mybot)

find_package(ament_cmake REQUIRED)
find_package(behaviortree_cpp_v3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_library(mybot_safe_diff_drive SHARED src/safe_diff_drive.cpp)
target_compile_features(mybot_safe_diff_drive PRIVATE cxx_std_17)
ament_target_dependencies(mybot_safe_diff_drive
  gazebo_dev gazebo_ros geometry_msgs nav_msgs rclcpp tf2_ros tf2_geometry_msgs)
target_link_libraries(mybot_safe_diff_drive ${GAZEBO_LIBRARIES})
install(TARGETS mybot_safe_diff_drive LIBRARY DESTINATION lib)

add_library(mybot_yield_pause_bt SHARED src/yield_pause_bt.cpp)
# BehaviorTree.CPP exports BT_RegisterNodesFromPlugin only with this definition.
target_compile_definitions(mybot_yield_pause_bt PRIVATE BT_PLUGIN_EXPORT)
target_compile_features(mybot_yield_pause_bt PRIVATE cxx_std_17)
ament_target_dependencies(mybot_yield_pause_bt behaviortree_cpp_v3 rclcpp std_msgs)
install(TARGETS mybot_yield_pause_bt LIBRARY DESTINATION lib)

ament_package()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
    PATTERN "setup_assistant.launch" EXCLUDE)
endif()

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/dashboard_bridge.py
  scripts/fixed_joint_pick_place.py
  scripts/moving_obstacles.py
  # Not a node: the predictive yield policy imported by fixed_joint_pick_place.
  # It is installed because the executor resolves it as a sibling module, and
  # under --symlink-install that only works if the install rule exists.
  scripts/obstacle_yield.py
  scripts/reset_dds_cache.sh
  scripts/rtf_probe.py
  # Re-stamps /scan with the current sim time so the Nav2 costmaps can raytrace
  # (the Gazebo lidar's own stamp is ~10 ms ahead of odom->base_footprint).
  scripts/scan_relay.py
  scripts/set_deepseek_key.sh
  scripts/start_competition.sh
  scripts/startup_gate.py
  scripts/system_health_check.py
  scripts/task_parser.py
  scripts/task_state_machine.py
  scripts/yield_velocity_gate.py
  scripts/yield_nav_profile.py
  DESTINATION lib/${PROJECT_NAME}
)
