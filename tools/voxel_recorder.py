#!/usr/bin/env python3
"""Record /local_costmap/voxel_grid (nav2_msgs/VoxelGrid) to JSONL.

The voxel grid is what actually keeps a stale mover mark alive: the 2D
costmap is re-projected from these voxels every cycle, so a mark in a
column the laser rays never cross survives even though no sensor sees
anything there.  Each record keeps the raw occupied-voxel list (x, y, z
index + cost) so it can be matched against a costmap snapshot later.
"""
import argparse
import base64
import json
import time
import zlib

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav2_msgs.msg import VoxelGrid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--rate', type=float, default=1.0)
    args = ap.parse_args()

    rclpy.init()
    node = Node('voxel_recorder')
    qos = QoSProfile(depth=2)
    qos.reliability = ReliabilityPolicy.RELIABLE
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    fh = open(args.out, 'w')
    state = {'last': -1e9}

    def cb(msg):
        now = time.time()
        if now - state['last'] < 1.0 / max(0.1, args.rate):
            return
        state['last'] = now
        occ = []
        for i, c in enumerate(msg.data):
            if c:
                occ.append([i, int(c)])
        rec = {
            't': now,
            'stamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'frame': msg.header.frame_id,
            'ox': msg.origin.position.x,
            'oy': msg.origin.position.y,
            'oz': msg.origin.position.z,
            'rx': msg.resolutions.x,
            'ry': msg.resolutions.y,
            'rz': msg.resolutions.z,
            'sx': msg.size_x,
            'sy': msg.size_y,
            'sz': msg.size_z,
            'n': len(occ),
            'occ': occ,
        }
        fh.write(json.dumps(rec) + '\n')
        fh.flush()

    node.create_subscription(VoxelGrid, '/local_costmap/voxel_grid', cb, qos)
    node.get_logger().info(f'voxel recorder -> {args.out}')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        fh.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
