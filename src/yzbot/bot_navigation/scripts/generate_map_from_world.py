#!/usr/bin/env python3

"""Rasterise a Gazebo world into a 2D occupancy map for Nav2/AMCL.

The map is generated straight from the world's collision geometry, so it always
matches the simulation: walls the lidar can hit are occupied cells, and dynamic
props (the small cubes) are excluded. This replaces hand-drawn or stale SLAM
maps whose features do not line up with the world, which is what makes AMCL
drift.

Usage:
    python3 generate_map_from_world.py <world> <output.pgm> [--resolution 0.05]
                                       [--margin 1.0] [--min-height 0.2]
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET


def local_pose(element):
    """Return (x, y, z, roll, pitch, yaw) from an SDF <pose>, or zeros."""
    pose = element.find('pose')
    if pose is None or not pose.text:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    values = [float(v) for v in pose.text.split()]
    return tuple(values + [0.0] * (6 - len(values)))


def compose(outer, inner):
    """Compose two planar poses (outer * inner)."""
    ox, oy, oz, orr, op, oyaw = outer
    ix, iy, iz, irr, ip, iyaw = inner
    cos_yaw, sin_yaw = math.cos(oyaw), math.sin(oyaw)
    return (
        ox + cos_yaw * ix - sin_yaw * iy,
        oy + sin_yaw * ix + cos_yaw * iy,
        oz + iz,
        orr + irr,
        op + ip,
        oyaw + iyaw,
    )


def collect_boxes(world_path, min_height, bounds=None):
    """Yield (x, y, size_x, size_y, yaw) for every tall-enough collision box.

    ``bounds`` = (x_min, y_min, x_max, y_max) keeps only boxes inside the
    operating area. Stray props elsewhere in the world cannot be seen from
    inside the room, so mapping them only bloats the costmap.
    """
    root = ET.parse(world_path).getroot()
    world = root.find('world')
    if world is None:
        raise ValueError(f'{world_path} has no <world> element')

    # A saved <state> block overrides poses at load time, and its link poses are
    # absolute world poses. Ignoring it silently shifts the whole map: this file
    # moves officeroom by +1.69 m in x, which is exactly the offset AMCL then
    # converges to, making every navigation goal land 1.7 m off.
    state_models = {}
    state = world.find('state')
    if state is not None:
        for model in state.findall('model'):
            state_models[model.get('name')] = {
                'pose': local_pose(model),
                'links': {
                    link.get('name'): local_pose(link)
                    for link in model.findall('link')
                },
            }

    boxes = []
    for model in world.findall('model'):
        override = state_models.get(model.get('name'))
        model_pose = override['pose'] if override else local_pose(model)
        for link in model.findall('link'):
            if override and link.get('name') in override['links']:
                link_pose = override['links'][link.get('name')]
            else:
                link_pose = compose(model_pose, local_pose(link))
            for collision in link.findall('collision'):
                size = collision.find('geometry/box/size')
                if size is None:
                    continue
                sx, sy, sz = (float(v) for v in size.text.split())
                pose = compose(link_pose, local_pose(collision))
                # Only obstacles tall enough for the 2D lidar to hit: this
                # drops the 3 cm cubes and the flat zone plates.
                if sz < min_height or pose[2] + sz < 0.15:
                    continue
                if bounds is not None:
                    bx0, by0, bx1, by1 = bounds
                    if not (bx0 <= pose[0] <= bx1 and by0 <= pose[1] <= by1):
                        continue
                boxes.append((pose[0], pose[1], sx, sy, pose[5]))
    return boxes


def half_extents(size_x, size_y, yaw):
    """True axis-aligned half extents of a rotated box (not its circumcircle)."""
    cos_yaw, sin_yaw = abs(math.cos(yaw)), abs(math.sin(yaw))
    return (
        0.5 * (cos_yaw * size_x + sin_yaw * size_y),
        0.5 * (sin_yaw * size_x + cos_yaw * size_y),
    )


def rasterise(boxes, resolution, margin):
    xs, ys = [], []
    for x, y, sx, sy, yaw in boxes:
        half_x, half_y = half_extents(sx, sy, yaw)
        xs.extend((x - half_x, x + half_x))
        ys.extend((y - half_y, y + half_y))
    if not xs:
        raise ValueError('no collision boxes found above the height threshold')
    x_min = math.floor((min(xs) - margin) / resolution) * resolution
    y_min = math.floor((min(ys) - margin) / resolution) * resolution
    x_max = math.ceil((max(xs) + margin) / resolution) * resolution
    y_max = math.ceil((max(ys) + margin) / resolution) * resolution
    width = int(round((x_max - x_min) / resolution))
    height = int(round((y_max - y_min) / resolution))
    # 254 = free, 0 = occupied. The image row 0 is the top of the map, i.e. the
    # highest y, so rows are flipped relative to world y.
    grid = [[254] * width for _ in range(height)]
    for x, y, sx, sy, yaw in boxes:
        half_x, half_y = sx / 2.0, sy / 2.0
        span_x, span_y = half_extents(sx, sy, yaw)
        cos_yaw, sin_yaw = math.cos(-yaw), math.sin(-yaw)
        col_lo = max(0, int((x - span_x - x_min) / resolution))
        col_hi = min(width - 1, int((x + span_x - x_min) / resolution) + 1)
        row_lo = max(0, int((y - span_y - y_min) / resolution))
        row_hi = min(height - 1, int((y + span_y - y_min) / resolution) + 1)
        for row in range(row_lo, row_hi + 1):
            world_y = y_min + (row + 0.5) * resolution
            for col in range(col_lo, col_hi + 1):
                world_x = x_min + (col + 0.5) * resolution
                dx, dy = world_x - x, world_y - y
                local_x = cos_yaw * dx - sin_yaw * dy
                local_y = sin_yaw * dx + cos_yaw * dy
                if abs(local_x) <= half_x and abs(local_y) <= half_y:
                    grid[height - 1 - row][col] = 0
    return grid, width, height, x_min, y_min


def write_pgm(path, grid, width, height):
    with open(path, 'wb') as handle:
        handle.write(f'P5\n# generated from world geometry\n{width} {height}\n255\n'.encode())
        handle.write(bytes(value for row in grid for value in row))


def write_yaml(path, image_name, resolution, origin):
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(f'image: {image_name}\n')
        handle.write('mode: trinary\n')
        handle.write(f'resolution: {resolution}\n')
        handle.write(f'origin: [{origin[0]:.3f}, {origin[1]:.3f}, 0.0]\n')
        handle.write('negate: 0\n')
        handle.write('occupied_thresh: 0.65\n')
        handle.write('free_thresh: 0.25\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('world', help='Path to the .world file')
    parser.add_argument('output', help='Output .pgm path (a .yaml sits beside it)')
    parser.add_argument('--resolution', type=float, default=0.05)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--min-height', type=float, default=0.2, dest='min_height')
    # A single comma-separated value avoids argparse mistaking negative
    # coordinates for options: pass --bounds=-16,-16,15,6.5
    parser.add_argument(
        '--bounds', default=None, metavar='X_MIN,Y_MIN,X_MAX,Y_MAX',
        help='Keep only obstacles inside this world-frame box.',
    )
    args = parser.parse_args(argv)
    bounds = None
    if args.bounds:
        parts = [float(v) for v in args.bounds.split(',')]
        if len(parts) != 4:
            parser.error('--bounds needs four comma-separated numbers')
        bounds = tuple(parts)

    boxes = collect_boxes(args.world, args.min_height, bounds)
    grid, width, height, x_min, y_min = rasterise(
        boxes, args.resolution, args.margin
    )
    write_pgm(args.output, grid, width, height)
    yaml_path = os.path.splitext(args.output)[0] + '.yaml'
    write_yaml(
        yaml_path, os.path.basename(args.output), args.resolution, (x_min, y_min)
    )
    occupied = sum(1 for row in grid for value in row if value == 0)
    print(f'boxes={len(boxes)} size={width}x{height} origin=({x_min:.2f},{y_min:.2f})')
    print(f'world x {x_min:.2f}..{x_min + width * args.resolution:.2f} '
          f'y {y_min:.2f}..{y_min + height * args.resolution:.2f}')
    print(f'occupied cells={occupied} ({100.0 * occupied / (width * height):.2f}%)')
    print(f'wrote {args.output} and {yaml_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
