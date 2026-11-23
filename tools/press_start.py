#!/usr/bin/env python3
"""Publish /competition/start for a few seconds so the latched press lands.

    python3 tools/press_start.py --seconds 10

Publishes std_msgs/Bool(data=True) at 5 Hz and reports how many subscriptions
were matched; exits on its own.
"""
import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--rate', type=float, default=5.0)
    args = parser.parse_args()

    rclpy.init()
    node = Node('press_start')
    # TRANSIENT_LOCAL + RELIABLE matches BOTH readers on the topic: the latched
    # one (which a VOLATILE publisher never reaches - measured 2026-11-23, the
    # parser stayed in WAITING_FOR_START for 6 min while the supervisor's
    # best-effort fallback did fire) and the best-effort fallback. Latched also
    # means a press made before a node subscribed still counts.
    qos = QoSProfile(depth=1)
    qos.reliability = ReliabilityPolicy.RELIABLE
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    pub = node.create_publisher(Bool, '/competition/start', qos)
    period = 1.0 / args.rate
    end = time.time() + args.seconds
    sent = 0
    while time.time() < end:
        msg = Bool()
        msg.data = True
        pub.publish(msg)
        sent += 1
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(period)
    subs = node.count_subscribers('/competition/start')
    node.get_logger().info(f'sent {sent} start messages, {subs} subscriber(s)')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
