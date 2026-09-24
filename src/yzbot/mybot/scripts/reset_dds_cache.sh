#!/usr/bin/env bash
# Reset the Fast DDS (rmw_fastrtps_cpp) leftovers that break discovery on WSL.
#
# Symptom this fixes: action goals are accepted by the controller but the
# response never arrives ("Failed to send goal response ... (timeout)"), or a
# freshly started node cannot see /navigate_to_pose, costmap services, etc.,
# even though the servers are alive. After many Ctrl-C / SIGKILL cycles the
# ros2 daemon and /dev/shm keep participants and fastrtps_* segments from dead
# processes, and new participants then fail to match endpoints.
#
# Usage (only while nothing is running):
#   ros2 run mybot reset_dds_cache.sh      # installed
#   bash src/yzbot/mybot/scripts/reset_dds_cache.sh
set -u

running=$(pgrep -a -f 'gzserver|gzclient|nav2_|controller_server|planner_server|bt_navigator|amcl|lifecycle_manager|robot_state_publisher|move_group|rviz2|fixed_joint_pick_place|task_state_machine|task_parser' | grep -v reset_dds_cache || true)
if [ -n "$running" ]; then
  echo "ROS/Gazebo processes are still running; stop them first:"
  echo "$running"
  exit 1
fi

echo "Stopping the ros2 daemon (if any)..."
ros2 daemon stop >/dev/null 2>&1 || true
pkill -f 'ros2cli.daemon.daemonize' >/dev/null 2>&1 || true
sleep 1

count=$(ls /dev/shm 2>/dev/null | grep -c '^fastrtps_' || true)
rm -f /dev/shm/fastrtps_*
echo "Removed ${count} stale Fast DDS shared-memory segment(s)."
echo "Remaining /dev/shm entries: $(ls /dev/shm 2>/dev/null | wc -l)"
