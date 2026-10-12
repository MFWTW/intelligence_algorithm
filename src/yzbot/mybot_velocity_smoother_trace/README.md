# Opt-in Nav2 Humble smoother telemetry (diagnostic only)

This package copies `nav2_velocity_smoother` v1.1.20 at Git tag
`a097086719c88f781aa59788eca29ac6ca5e56db` (Apache-2.0).
The copied header was byte-identical to the installed Humble 1.1.20 header
(SHA-256 `0c1c4f3b1609e9eddb4a2f3c6029ceff6b1247cffaf7fb0725e010fb75aaaedd`).
It installs under **a different package name** and never shadows the stock
`nav2_velocity_smoother` package or binary. The default competition launch
continues to include stock `nav2_bringup/bringup_launch.py`.

When explicitly enabled in the isolated simulation by
`smoother_trace_experiment:=true smoother_trace_file:=/absolute/path.jsonl`, a
version-pinned wrapper constructs the same 8 Nav2 nodes and changes only the
package/executable of its one velocity smoother. It preserves the stock
`velocity_smoother` node name, lifecycle, RELIABLE/depth-1 input QoS, topic
remaps, params, rate, kinematic limits, timeout, smoothing algorithm and
output publication. A missing/unwritable trace path fails before navigation
activation. This mode is mutually exclusive with `smoother_udp_experiment`;
all participants keep the stock DDS transport. A proposed BEST_EFFORT input
QoS candidate was **withdrawn without any simulation** after static review:
losing the single handoff stop command could permit continued base motion
until timeout and subsequent deceleration. The trace binary does not expose
that option.

Instrumentation increments atomic counters at callback entry/valid/invalid,
20-Hz timer entry/no-command/timeout-stopped/timeout-deceleration, and before/
after `publish`. A scope guard also records each timer callback's last duration
and the maximum duration in each one-second writer interval, plus completed
callback count and age of any currently in-flight callback. A brief timeout
return versus a scheduler-heavy timer can be compared without changing any
control command. The writer samples these atomics independently: **last and
interval maximum need not refer to the same callback/second**; a final peak
may be unflushed when the lifecycle stops. Neither an empty lost-message
event nor a fast timer proves whether RMW took an upstream sample. It records
the last monotonic times, ROS clock nanoseconds,
last valid angular input/output (milliradians/sec), linear input/output
(millimeters/sec), process PID (to detect respawn), lifecycle state, and the
`FASTRTPS_DEFAULT_PROFILES_FILE` this process actually started with (read-only,
`null` when it got none). That last field is what lets a UDP-transport round be
told apart from a default-transport one without trusting any external observer.
A **separate non-ROS, wall-time writer thread** flushes one JSONL row/second;
if the ROS executor stops servicing both callback and timer, the writer can
still report stationary counters. It never publishes Twist, creates no ROS
subscription or client, and cannot establish whether DDS delivered a `publish`
call's payload. Its extra atomics, wall-clock reads, writer thread and changed
binary can still perturb scheduling; this is **diagnostic, not a speed A/B**.

Decision table for paired observer + internal monotonic time:

| During fresh upstream requests | Strongest supported inference |
| --- | --- |
| Internal `input_valid` flat | Smoother subscriber DDS take, executor delivery, or upstream endpoint mismatch; not necessarily publisher fault. |
| `input_valid` flat **and `poll_taken` rises** | The reader DID hold a sample the executor never dispatched: waitset/executor delivery fault, not DDS loss. |
| `input_valid` flat, `poll_attempts` rising, **`poll_taken` stays 0** | Nothing was in the reader at those instants: the loss is below rclcpp, in DDS/RMW delivery. |
| `poll_attempts` stays 0 | The probe never engaged (its gate needs a prior command), so it proves nothing either way. |

`MYBOT_SMOOTHER_POLL_FALLBACK_SEC` arms that probe. It is **strictly diagnostic**:
on a stalled tick the timer asks the reader directly with `take()`, counts what it
finds, and **discards it** - it never calls `inputCommandCallback`. An earlier
version fed the sample back into the control path; that was unsafe, because the
callback re-stamps `last_command_time_` and would cancel the zero-Twist timeout
stop, letting the base resume on a command up to a minute old whose staleness is
undetectable (Twist has no header). Discarding can only leave the smoother in the
state it was already in, so `poll_taken` measures the reader without moving the
robot. Because it consumes that sample, an armed round is still a perturbed
observation: treat `poll_*` as diagnostic and never as a speed A/B.
| Input rises, `timer_enter` flat | Wall timer/executor scheduling fault. |
| Input rises, timer rises, `timer_timeout_stopped` rises | Compare recorded ROS-clock/input times; old/missing valid commands or time discontinuity. |
| Input & timer rise, `publish_after` flat | Internal early return or exception; inspect counters and logs. |
| `publish_after` rises, two independent `/cmd_vel` probes flat | Output DDS delivery/observation issue, not proof of wheel command delivery. |

The runner first refuses to run if any competition/Gazebo/Nav2 process exists,
because its legacy `--kill` and cleanup otherwise terminate the **whole ROS
host**, including any other user's stack. Only run on an exclusively allocated
WSL/ROS environment; a concurrent stack created after preflight is still at
risk. Its offline verifier fails trace coverage for a >3.5 s row gap, a changed
PID, flat callback/timer/publish totals, a stock smoother launched alongside
the trace smoother, or a cleanup that leaves a process. Store both prelaunch and
postcleanup source/binary hash manifests to avoid retrospective drift.

Keep smoother, cancellation, attached-object and placement safety gates unchanged.
This experiment does **not** authorize production use or a speedup claim.
