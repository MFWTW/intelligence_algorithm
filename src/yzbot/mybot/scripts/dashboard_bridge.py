#!/usr/bin/env python3
"""Publish the scoring-dashboard topics required by the competition rules.

The rule sheet asks each team to publish standard ROS topics so the judges'
Foxglove layout can show, live:

    任务信息：任务题目、放置区域、每种资源抓取的数量
    当前状态：任务进度（当前抓到的第几个）、工作状态、末端状态（抓取/放置）

Everything already exists, but as *JSON strings* on ``/competition/task`` and
``/competition/progress`` - and a Foxglove Indicator panel can only read a
numeric/string field of a topic, it cannot index into a JSON string. So this
node derives small, directly plottable topics:

    /competition/dashboard/red_count      std_msgs/Int32   红色需要抓几个
    /competition/dashboard/blue_count     std_msgs/Int32   蓝色需要抓几个
    /competition/dashboard/red_zone       std_msgs/String  "红色到A区"
    /competition/dashboard/blue_zone      std_msgs/String  "蓝色到C区"
    /competition/dashboard/grab_index     std_msgs/Int32   当前第几个（1 起）
    /competition/dashboard/total          std_msgs/Int32   总数
    /competition/dashboard/completed      std_msgs/Int32   已完成数量
    /competition/dashboard/work_state     std_msgs/String  中文工作状态
    /competition/dashboard/gripper_state  std_msgs/String  末端状态：未抓取/已抓取/已释放
    /competition/dashboard/current_cube   std_msgs/String  red_cube_2
    /competition/dashboard/problem        std_msgs/String  任务题目原文

All of them are transient-local (latched), so opening Foxglove in the middle of
a run still shows the current values. This node only *reads* the supervisor
topics - it never publishes anything the controller consumes.
"""

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String


# Supervisor states -> what a judge should read on the dashboard.
WORK_STATES = {
    'IDLE': '待命',
    'PARSING_TASK': '解析任务',
    'VALIDATING_TASK': '校验任务',
    'DETECTING_OBJECT': '识别资源',
    'NAV_TO_OBJECT': '前往资源点',
    'PRE_GRASP': '对准资源',
    'GRASPING': '执行抓取',
    'VERIFY_GRASP': '已抓取',
    'NAV_TO_ZONE': '前往放置区',
    'PLACING': '执行放置',
    'VERIFY_PLACE': '校验放置',
    'NEXT_OBJECT': '下一块',
    'COMPLETED': '全部完成',
    'FAILED': '任务失败',
}

# End-effector state, from the single-object executor (finer than the supervisor).
GRIPPER_STATES = {
    'PRE_GRASP': '未抓取',
    'NAV_TO_OBJECT': '未抓取',
    'DETECTING_OBJECT': '对准中',
    'GRASPING': '抓取中',
    'VERIFY_GRASP': '已抓取',
    'NAV_TO_ZONE': '携带中',
    'PLACING': '放置中',
    'VERIFY_PLACE': '已释放',
}

COLOUR_LABEL = {'red': '红色', 'blue': '蓝色'}


class DashboardBridge(Node):
    """Fan the supervisor's JSON strings out into dashboard-friendly topics."""

    def __init__(self) -> None:
        super().__init__('dashboard_bridge')
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self._pubs = {
            name: self.create_publisher(msg_type, f'/competition/dashboard/{name}', qos)
            for name, msg_type in (
                ('red_count', Int32),
                ('blue_count', Int32),
                ('red_zone', String),
                ('blue_zone', String),
                ('grab_index', Int32),
                ('total', Int32),
                ('completed', Int32),
                ('work_state', String),
                ('gripper_state', String),
                ('current_cube', String),
                ('problem', String),
            )
        }

        self.create_subscription(String, '/competition/task', self._task_cb, qos)
        self.create_subscription(
            String, '/competition/task_problem', self._problem_cb, qos
        )
        self.create_subscription(String, '/competition/state', self._state_cb, qos)
        self.create_subscription(
            String, '/competition/executor_state', self._executor_cb, qos
        )
        self.create_subscription(String, '/competition/progress', self._progress_cb, qos)
        self.create_subscription(
            String, '/competition/current_object', self._object_cb, qos
        )

        # Keep the latched values fresh for late subscribers.
        self.create_timer(2.0, self._republish)
        self.get_logger().info(
            'Dashboard bridge ready: /competition/dashboard/* for the scoring '
            'layout (see FOXGLOVE.md).'
        )

    # ------------------------------------------------------------- helpers
    def _int(self, name: str, value: int) -> None:
        msg = Int32()
        msg.data = int(value)
        self._pubs[name].publish(msg)

    def _str(self, name: str, value: str) -> None:
        msg = String()
        msg.data = str(value)
        self._pubs[name].publish(msg)

    # ---------------------------------------------------------- callbacks
    def _task_cb(self, msg: String) -> None:
        """Structured task -> per-colour counts and target zones."""
        try:
            payload = json.loads(msg.data)
            groups = payload['tasks']
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        for group in groups:
            colour = str(group.get('color', '')).lower()
            if colour not in COLOUR_LABEL:
                continue
            label = COLOUR_LABEL[colour]
            count = int(group.get('count', 0))
            zone = str(group.get('zone', '')).upper()
            self._int(f'{colour}_count', count)
            self._str(f'{colour}_zone', f'{label}到{zone}区' if zone else '')
        total = sum(int(g.get('count', 0)) for g in groups)
        self._int('total', total)
        self.get_logger().info(
            f'Dashboard task: {total} objects '
            f'({", ".join(str(g) for g in groups)})'
        )

    def _problem_cb(self, msg: String) -> None:
        text = msg.data.strip()
        # The generator prints "题目: ..." itself; avoid showing it twice.
        for prefix in ('题目:', '题目：'):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        self._str('problem', text)

    def _state_cb(self, msg: String) -> None:
        state = msg.data.strip()
        self._str('work_state', WORK_STATES.get(state, state or '待命'))

    def _executor_cb(self, msg: String) -> None:
        state = msg.data.strip()
        self._str('gripper_state', GRIPPER_STATES.get(state, state or '未抓取'))

    def _progress_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        total = int(payload.get('total', 0))
        completed = int(payload.get('completed', 0))
        index = int(payload.get('current_index', 0))
        self._int('total', total)
        self._int('completed', completed)
        # "第几个" is 1-based and never exceeds the queue length.
        self._int('grab_index', min(index + 1, total) if total else 0)

    def _object_cb(self, msg: String) -> None:
        self._str('current_cube', msg.data.strip())

    def _republish(self) -> None:
        """Nothing to do: values are latched. Kept for a periodic heartbeat log."""
        self.get_logger().debug('dashboard heartbeat')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DashboardBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
