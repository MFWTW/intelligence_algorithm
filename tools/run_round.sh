#!/usr/bin/env bash
# Run one competition round end to end and keep every artefact in <out_dir>.
#
#   tools/run_round.sh <mapping_rule> <out_dir>
#
# Steps: write mapping_rule -> clean leftovers -> start the stack detached ->
# wait for "系统就绪" -> start tools/round_recorder.py -> publish
# /competition/start -> wait for a terminal state -> stop the whole process
# group -> verify nothing is left behind.
set -uo pipefail

RULE=${1:?usage: run_round.sh <mapping_rule> <out_dir>}
OUT=${2:?usage: run_round.sh <mapping_rule> <out_dir>}
ROOT=/home/xu/intelligence_algorithm-main
CFG="$ROOT/src/yzbot/mybot/config/task_parser.yaml"
PATTERN='gzserv[e]r|[/ ]nav2[_]|move_grou[p]|foxglove_bridg[e]|task_state_machin[e]|task_parse[r]|fixed_joint_pick_plac[e]|ros2 launc[h]|robot_state_publishe[r]|system_health_chec[k]|rviz[2]'

mkdir -p "$OUT"
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "== round mapping_rule=$RULE -> $OUT"

# ---------------------------------------------------------------- 1. rule
python3 - "$CFG" "$RULE" <<'PY'
import re
import sys
path, rule = sys.argv[1], sys.argv[2]
text = open(path, encoding='utf-8').read()
new, n = re.subn(r'(?m)^(\s*mapping_rule:\s*).*$',
                 lambda m: m.group(1) + rule, text)
if n != 1:
    raise SystemExit(f'mapping_rule occurrences: {n}')
open(path, 'w', encoding='utf-8').write(new)
print(f'mapping_rule -> {rule}')
PY
grep -n 'mapping_rule' "$CFG" | tee "$OUT/rule.txt"

# ------------------------------------------------------ 2. problem text
( cd "$ROOT" && timeout 90 ./TMSCQtest_x86_x64.bin 11470507 ) \
    > "$OUT/problem.txt" 2>&1
say "problem: $(tr '\n' ' ' < "$OUT/problem.txt" | cut -c1-160)"

# --------------------------------------------------------- 3. clean up
say "cleaning leftovers"
pkill -f 'gzserv[e]r' 2>/dev/null
pkill -f '[/ ]nav2[_]' 2>/dev/null
pkill -f 'move_grou[p]' 2>/dev/null
pkill -f 'foxglove_bridg[e]' 2>/dev/null
pkill -f 'task_state_machin[e]' 2>/dev/null
pkill -f 'task_parse[r]' 2>/dev/null
pkill -f 'fixed_joint_pick_plac[e]' 2>/dev/null
pkill -f 'ros2 launc[h]' 2>/dev/null
pkill -f 'robot_state_publishe[r]' 2>/dev/null
pkill -f 'system_health_chec[k]' 2>/dev/null
pkill -f 'rviz[2]' 2>/dev/null
for _ in $(seq 1 20); do
    pgrep -f "$PATTERN" >/dev/null 2>&1 || break
    sleep 1
done
pgrep -a -f "$PATTERN" | grep -v start_competition > "$OUT/leftovers_before.txt" || true

# ----------------------------------------------------------- 4. launch
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$ROOT/install/setup.bash"
set -u
export RCUTILS_LOGGING_BUFFERED_STREAM=1
export MYBOT_MOVING_OBSTACLES=${MYBOT_MOVING_OBSTACLES:-true}
# ~/.ros is outside the writable workspace, so keep every ROS artefact here.
export ROS_HOME="$OUT/roshome"
export ROS_LOG_DIR="$OUT/roslog"
mkdir -p "$ROS_HOME" "$ROS_LOG_DIR"

say "launching"
setsid bash -c 'exec ros2 run mybot start_competition.sh' \
    > "$OUT/stack.log" 2>&1 &
STACK_PID=$!
say "stack pid=$STACK_PID"
echo "$STACK_PID" > "$OUT/stack.pid"
sleep 1
ps -o pid,pgid,args -p "$STACK_PID" >> "$OUT/stack.pid"

READY=0
for i in $(seq 1 300); do
    if grep -qE '系统就绪|SYSTEM READY' "$OUT/stack.log" 2>/dev/null; then
        READY=1
        break
    fi
    if ! kill -0 "$STACK_PID" 2>/dev/null; then
        say "stack process exited during startup"
        break
    fi
    sleep 1
done
if [ "$READY" != 1 ]; then
    say "READY FAILED; tail of log:"
    tail -40 "$OUT/stack.log"
    kill -INT -"$STACK_PID" 2>/dev/null
    sleep 5
    kill -TERM -"$STACK_PID" 2>/dev/null
    exit 2
fi
say "system ready after ${i}s"

# ------------------------------------------------------ 4b. graph diagnostics
say "graph diagnostics"
{
    echo "--- node count: $(timeout 20 ros2 node list 2>/dev/null | wc -l)"
    echo "--- duplicate node names:"
    timeout 20 ros2 node list 2>/dev/null | sort | uniq -d
    echo "--- arm action:"
    timeout 20 ros2 action info /arm_controller/follow_joint_trajectory 2>&1 | head -20
    echo "--- gripper action:"
    timeout 20 ros2 action info /gripper_controller/follow_joint_trajectory 2>&1 | head -20
} > "$OUT/graph.txt" 2>&1

# ------------------------------------------- 4c. one-goal duplication probe
# 2026-11-23: rounds 11d-11k showed every FollowJointTrajectory goal logged
# twice by the controller (status=5 CANCELED) although the client sent once.
# This probe sends exactly one goal and counts the resulting
# "Received new action goal" lines, separating "client sent twice" from
# "one write, two servers".
if [ "${ARM_GOAL_PROBE:-0}" = 1 ]; then
    say "arm goal probe"
    before=$(grep -c 'Received new action goal' "$OUT/stack.log" 2>/dev/null || true)
    timeout 90 python3 "$ROOT/tools/arm_goal_probe.py" \
        > "$OUT/arm_goal_probe.log" 2>&1
    probe_rc=$?
    sleep 2
    after=$(grep -c 'Received new action goal' "$OUT/stack.log" 2>/dev/null || true)
    {
        echo "=== arm goal probe rc=$probe_rc"
        echo "received_before=$before received_after=$after"
        echo "received_delta=$((after - before))  (1 send; 2 means two servers)"
        cat "$OUT/arm_goal_probe.log"
        echo "--- arm action after probe:"
        timeout 20 ros2 action info /arm_controller/follow_joint_trajectory 2>&1 | head -20
        echo "--- gripper action after probe:"
        timeout 20 ros2 action info /gripper_controller/follow_joint_trajectory 2>&1 | head -20
    } >> "$OUT/graph.txt" 2>&1
    say "arm goal probe delta=$((after - before))"
fi

if [ "${ARM_ONLY:-0}" = 1 ]; then
    say "ARM_ONLY set: stopping after the probe"
    {
        echo "=== controllers before shutdown"
        timeout 20 ros2 control list_controllers 2>&1 | head -10
    } >> "$OUT/graph.txt" 2>&1
    kill -INT -"$STACK_PID" 2>/dev/null
    sleep 5
    kill -TERM -"$STACK_PID" 2>/dev/null
    exit 0
fi

# ---------------------------------------------------------- 5. record
say "starting recorder"
python3 "$ROOT/tools/round_recorder.py" --out "$OUT/events.jsonl" \
    --max-duration 1500 --after-terminal 10 > "$OUT/recorder.log" 2>&1 &
REC_PID=$!
PROBE_PID=""
if [ "${COLLISION_PROBE:-0}" = 1 ]; then
    say "starting collision probe (local costmap + arc + scan)"
    python3 "$ROOT/tools/collision_probe.py" --out "$OUT/probe.jsonl" \
        --max-duration 1500 --after-terminal 10 > "$OUT/probe.log" 2>&1 &
    PROBE_PID=$!
fi
sleep 4
if [ "${START_DELAY:-0}" -gt 0 ]; then
    say "holding the start gate for ${START_DELAY}s to shift mover phase"
    sleep "${START_DELAY}"
fi
say "publishing /competition/start"
timeout 30 python3 "$ROOT/tools/press_start.py" --seconds 10 \
    > "$OUT/start_pub.log" 2>&1
say "start published (rc=$?)"
# The press has to cross DDS to a latched subscriber; retry once if the
# state machine never left WAITING_FOR_START.
for _ in $(seq 1 10); do
    [ "$(grep -c 'State -> ' "$OUT/stack.log")" -ge 2 ] && break
    sleep 2
done
if [ "$(grep -c 'State -> ' "$OUT/stack.log")" -lt 2 ]; then
    say "start press did not land, retrying"
    timeout 30 python3 "$ROOT/tools/press_start.py" --seconds 10 \
        >> "$OUT/start_pub.log" 2>&1
fi
# Second graph capture: by now every server (gzserver's arm/gripper JTC
# included) has finished loading, so a duplicate server is visible.
sleep 8
{
    echo "=== graph diagnostics after start ==="
    echo "--- node count: $(timeout 20 ros2 node list 2>/dev/null | wc -l)"
    echo "--- duplicate node names:"
    timeout 20 ros2 node list 2>/dev/null | sort | uniq -d
    echo "--- arm action:"
    timeout 20 ros2 action info /arm_controller/follow_joint_trajectory 2>&1 | head -24
} >> "$OUT/graph.txt" 2>&1

# --------------------------------------------------------- 6. wait end
DONE=0
for i in $(seq 1 700); do
    if grep -qE 'State -> (COMPLETED|FAILED|ABORTED)' "$OUT/stack.log"; then
        DONE=1; break
    fi
    if ! kill -0 "$REC_PID" 2>/dev/null; then
        say "recorder stopped early"
        break
    fi
    sleep 2
done
if [ "$DONE" = 1 ]; then
    say "terminal state after ~$((i * 2))s: $(grep -oE 'State -> (COMPLETED|FAILED|ABORTED)' "$OUT/stack.log" | tail -1)"
else
    say "no terminal state seen; check log"
fi
sleep 12

# ------------------------------------------------------------- 7. stop
say "stopping stack (SIGINT to process group $STACK_PID)"
kill -INT -"$STACK_PID" 2>/dev/null
for _ in $(seq 1 30); do
    pgrep -f "$PATTERN" >/dev/null 2>&1 || break
    sleep 1
done
if pgrep -f "$PATTERN" >/dev/null 2>&1; then
    say "still running after 30s, SIGTERM"
    kill -TERM -"$STACK_PID" 2>/dev/null
    sleep 5
    pkill -9 -f 'gzserv[e]r' 2>/dev/null
    pkill -9 -f '[/ ]nav2[_]' 2>/dev/null
    pkill -9 -f 'move_grou[p]' 2>/dev/null
    pkill -9 -f 'fixed_joint_pick_plac[e]' 2>/dev/null
    pkill -9 -f 'task_state_machin[e]' 2>/dev/null
    pkill -9 -f 'task_parse[r]' 2>/dev/null
fi
for pid in "$REC_PID" "$PROBE_PID"; do
    [ -n "$pid" ] && kill -INT "$pid" 2>/dev/null && wait "$pid" 2>/dev/null || true
done
pgrep -a -f "$PATTERN" | grep -v start_competition > "$OUT/leftovers_after.txt" || true
say "leftovers after: $(wc -l < "$OUT/leftovers_after.txt")"

# ------------------------------------------------------------ 8. summary
say "TIMING lines: $(grep -c 'TIMING ' "$OUT/stack.log" || true)   resident goals: $(grep -c 'Resident goal' "$OUT/stack.log" || true)   exec results: $(grep -c 'Executor result' "$OUT/stack.log" || true)   YIELD: $(grep -cE 'YIELD (go|cap|arc|stop)' "$OUT/stack.log" || true)"
say "events: $(wc -l < "$OUT/events.jsonl" 2>/dev/null || echo 0) lines"
say "done"
