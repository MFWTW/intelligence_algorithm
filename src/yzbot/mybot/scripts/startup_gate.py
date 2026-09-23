#!/usr/bin/env python3

"""Exit successfully once Gazebo, sensors, and ros2_control are ready."""

import sys
import time

import rclpy
from controller_manager_msgs.srv import ListControllers
from rclpy.node import Node


class StartupGate(Node):
    def __init__(self) -> None:
        super().__init__('startup_gate')
        self.declare_parameter('timeout_sec', 180.0)
        self._deadline = time.monotonic() + float(
            self.get_parameter('timeout_sec').value
        )
        self._client = self.create_client(
            ListControllers, '/controller_manager/list_controllers'
        )
        self._future = None
        self.done = False
        self.success = False
        self._timer = self.create_timer(0.5, self._tick)
        self.get_logger().info(
            'Waiting for Gazebo sensors and all ros2_control controllers.'
        )

    def _tick(self) -> None:
        if self.done:
            return
        if time.monotonic() >= self._deadline:
            self.get_logger().error('Startup prerequisite gate timed out.')
            self.done = True
            return

        topics = {name for name, _ in self.get_topic_names_and_types()}
        required_topics = {
            '/clock', '/joint_states', '/odom', '/scan', '/tf', '/tf_static',
            '/camera/image_raw',
        }
        if required_topics - topics or not self._client.service_is_ready():
            return

        if self._future is None:
            self._future = self._client.call_async(ListControllers.Request())
            return
        if not self._future.done():
            return
        try:
            response = self._future.result()
        except Exception as exc:
            self.get_logger().warning(f'Controller readiness query failed: {exc}')
            self._future = None
            return
        self._future = None
        active = {
            controller.name
            for controller in response.controller
            if controller.state == 'active'
        }
        required = {
            'joint_state_broadcaster', 'arm_controller', 'gripper_controller'
        }
        if required <= active:
            self.get_logger().info(
                'STARTUP PREREQUISITES READY: starting MoveIt and Nav2.'
            )
            self.success = True
            self.done = True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StartupGate()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    success = node.success
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
