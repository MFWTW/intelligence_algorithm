#!/usr/bin/env bash
# Stable launcher for the competition stack on this WSL host.
#
#   ros2 run mybot start_competition.sh              # 推荐：无 GUI，自动预检+清缓存+等就绪
#   ros2 run mybot start_competition.sh --gui        # 需要 Gazebo 窗口时
#   ros2 run mybot start_competition.sh --with-rviz  # 需要 MoveIt RViz 时
#   ros2 run mybot start_competition.sh --udp        # 实验性：纯 UDP 传输（见下方注意事项）
#   ros2 run mybot start_competition.sh --kill       # 先自动清掉残留进程再启动
#
# Why a wrapper instead of a bare `ros2 launch`:
#
#   * Every Ctrl-C / kill leaves Fast DDS shared-memory segments behind; after a
#     few cycles new nodes randomly fail to match endpoints and the bringup dies
#     with "Switch controller timed out after 5.000000 seconds" or a costmap that
#     waits forever for map -> base_footprint. The wrapper clears them first.
#   * A killed launch often leaves orphan Nav2/gzserver processes running. A
#     second stack then fights over /cmd_vel and the robot never moves, so the
#     wrapper refuses to start until the leftovers are gone.
#   * Gazebo's GUI roughly doubles the render load (measured RTF 0.18 with the
#     full stack), so the GUI is opt-in here and Foxglove is the default view.
#   * It waits for /competition/system_ready and prints what is missing instead
#     of leaving you with a silent terminal and a robot that never moves.
#
# --udp 注意：本机实测纯 UDP 会把 bringup 卡在 spawn_entity —— 它一直等
# latched 的 /robot_description（大消息的历史样本没能送达），机器人不会生成。
# 只有在共享内存彻底不可用时才试它，并且要盯住 "Successfully spawned entity"。
set -uo pipefail

WS="${MYBOT_WS:-$HOME/dev_ws}"
USE_GUI=false
USE_RVIZ=false
USE_UDP=false
KILL_FIRST=false
READY_TIMEOUT=300
LAUNCH_ARGS=()

usage() {
  # Print the leading comment block (line 2 until the first non-comment line).
  awk 'NR > 1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "$0"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --gui) USE_GUI=true ;;
    --with-rviz) USE_RVIZ=true ;;
    --udp) USE_UDP=true ;;
    --kill) KILL_FIRST=true ;;
    --wait) READY_TIMEOUT="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) LAUNCH_ARGS+=("$1") ;;
  esac
  shift
done

LEFT_OVER_PATTERN='gzserv[e]r|nav2[_]|move_grou[p]|foxglove_bridg[e]|task_state_machin[e]|task_parse[r]|fixed_joint_pick_plac[e]|ros2 launc[h]|robot_state_publishe[r]|system_health_chec[k]|rviz[2]'

echo "== 1/5 预检残留进程 =="
leftover=$(pgrep -a -f "$LEFT_OVER_PATTERN" 2>/dev/null | grep -v "start_competition" || true)
if [ -n "$leftover" ]; then
  echo "发现上一次的残留进程："
  echo "$leftover" | sed 's/^/    /'
  if [ "$KILL_FIRST" = true ]; then
    echo "  --kill 已指定，正在清理..."
    pkill -f 'nav2[_]' 2>/dev/null
    pkill -f 'gzserv[e]r' 2>/dev/null
    pkill -f 'move_grou[p]' 2>/dev/null
    pkill -f 'foxglove_bridg[e]' 2>/dev/null
    pkill -f 'task_state_machin[e]' 2>/dev/null
    pkill -f 'task_parse[r]' 2>/dev/null
    pkill -f 'fixed_joint_pick_plac[e]' 2>/dev/null
    pkill -f 'ros2 launc[h]' 2>/dev/null
    pkill -f 'robot_state_publishe[r]' 2>/dev/null
    pkill -f 'system_health_chec[k]' 2>/dev/null
    pkill -f 'rviz[2]' 2>/dev/null
    sleep 6
  else
    echo
    echo "两套 Nav2 会抢 /cmd_vel，机器人不会动。请任选："
    echo "    ros2 run mybot start_competition.sh --kill      # 自动清理后启动"
    echo "    pkill -f 'nav2_'; pkill -f 'gzserv[e]r'; pkill -f 'move_grou[p]'"
    exit 1
  fi
fi

echo "== 2/5 编译结果与环境 =="
if [ ! -f "$WS/install/setup.bash" ]; then
  echo "找不到 $WS/install/setup.bash，请先 colcon build。" >&2
  exit 1
fi
# ROS's setup.bash reads several variables without defaults, so nounset (set -u)
# must be off while sourcing it.
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
set -u
if ! ros2 pkg prefix mybot >/dev/null 2>&1; then
  echo "环境里没有 mybot 包，检查 $WS/install 是否编译成功。" >&2
  exit 1
fi

echo "== 3/5 清理 Fast DDS 残留 =="
bash "$WS/src/yzbot/mybot/scripts/reset_dds_cache.sh" || true

if [ "$USE_UDP" = true ]; then
  export FASTRTPS_DEFAULT_PROFILES_FILE="$WS/src/yzbot/mybot/config/fastdds_wsl_udp.xml"
  echo "== 传输方式：纯 UDP（$FASTRTPS_DEFAULT_PROFILES_FILE）=="
else
  echo "== 传输方式：默认共享内存（若连续抽风，下次加 --udp）=="
fi

LAUNCH_CMD=(ros2 launch mybot competition_bringup.launch.py
  "gui:=$USE_GUI" "moveit_rviz:=$USE_RVIZ" "parser_linger_sec:=600")
if [ ${#LAUNCH_ARGS[@]} -gt 0 ]; then
  LAUNCH_CMD+=("${LAUNCH_ARGS[@]}")
fi

LOG=$(mktemp -t mybot_start_XXXX.log)
echo "== 4/5 启动 =="
echo "    ${LAUNCH_CMD[*]}"
echo "    日志：$LOG"
echo "    （Ctrl-C 会一起停掉整套，避免留下孤儿进程）"

"${LAUNCH_CMD[@]}" 2>&1 | tee "$LOG" &
LAUNCH_PID=$!

cleanup() {
  echo
  echo "正在停止整套（Ctrl-C）..."
  kill -INT "$LAUNCH_PID" 2>/dev/null
  wait "$LAUNCH_PID" 2>/dev/null
  sleep 5
  still=$(pgrep -a -f "$LEFT_OVER_PATTERN" 2>/dev/null | grep -v "start_competition" || true)
  if [ -n "$still" ]; then
    echo "仍有残留进程，请执行："
    echo "    pkill -f 'nav2_'; pkill -f 'gzserv[e]r'; pkill -f 'move_grou[p]'"
    echo "    ros2 run mybot reset_dds_cache.sh"
  else
    echo "已干净退出。下次启动前建议再跑一次 reset_dds_cache.sh。"
  fi
  exit 130
}
trap cleanup INT TERM

echo "== 5/5 等待 /competition/system_ready（最多 ${READY_TIMEOUT}s）=="
python3 - "$READY_TIMEOUT" <<'PY' &
import sys, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Bool, String

timeout = float(sys.argv[1])
qos = QoSProfile(depth=1)
qos.reliability = ReliabilityPolicy.RELIABLE
qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

rclpy.init()
node = Node('bringup_watchdog')
state = {'ready': None, 'health': ''}
node.create_subscription(Bool, '/competition/system_ready', lambda m: state.update(ready=bool(m.data)), qos)
node.create_subscription(String, '/competition/system_health', lambda m: state.update(health=m.data), qos)

import time
deadline = time.monotonic() + timeout
last = None
while rclpy.ok() and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.2)
    if state['ready'] != last:
        last = state['ready']
        if last is True:
            print('\n✅ 系统就绪：SYSTEM READY —— 状态机会自动开始第一个方块。', flush=True)
            print('   Foxglove 连 ws://localhost:8765 看进度。', flush=True)
            break
        if last is False and state['health']:
            print(f"\n⏳ 尚未就绪：{state['health'][:220]}", flush=True)
if state['ready'] is not True:
    print(f"\n⚠️  {timeout:.0f}s 内没有等到 system_ready。", flush=True)
    print("   排查顺序：Switch controller timed out / TF 报错 / 话题缺失 →", flush=True)
    print("   停掉整套，然后 ros2 run mybot start_competition.sh --kill --udp", flush=True)
node.destroy_node()
rclpy.shutdown()
PY
WATCH_PID=$!

wait "$LAUNCH_PID"
kill "$WATCH_PID" 2>/dev/null
echo "启动进程已退出（日志：$LOG）。"
