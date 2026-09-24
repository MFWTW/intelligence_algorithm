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
  scripts/fixed_joint_pick_place.py
  scripts/reset_dds_cache.sh
  scripts/startup_gate.py
  scripts/system_health_check.py
  scripts/task_parser.py
  scripts/task_state_machine.py
  DESTINATION lib/${PROJECT_NAME}
)
