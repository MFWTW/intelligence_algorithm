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
# The shared-memory *segment* is only half of a Fast DDS port: each port also
# owns a named semaphore "sem.fastrtps_port<port>_mutex". Deleting the segment
# while leaving the semaphore behind is exactly what produces
#   [RTPS_TRANSPORT_SHM Error] Failed init_port fastrtps_port74xx:
#   open_and_lock_file failed
# on the next start, which then cascades into
#   CRITICAL FAILURE: SERVER map_server IS DOWN
# (9-round campaign 2026-11-16/17: r5/r7/r7b/r9b failed to start with 16/8/21/15
# such errors, while all 9 rounds that started cleanly had zero - no exception
# in 14 launches). Remove both kinds together.
semcount=$(ls /dev/shm 2>/dev/null | grep -c '^sem\.fastrtps_port' || true)
rm -f /dev/shm/sem.fastrtps_port*
echo "Removed ${count} stale Fast DDS shared-memory segment(s) and ${semcount} port semaphore(s)."
echo "Remaining /dev/shm entries: $(ls /dev/shm 2>/dev/null | wc -l)"
