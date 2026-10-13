cmake_minimum_required(VERSION 3.22)
project(mybot)

find_package(ament_cmake REQUIRED)

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
  DESTINATION lib/${PROJECT_NAME}
)
