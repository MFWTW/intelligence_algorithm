#!/usr/bin/env python3
"""Draw the world's object positions onto the navigation map.

Answers "如果我把搬运物块的位置改了，它在地图上到底在哪？" - the world file uses
metres, the map is a PGM whose origin/resolution live in the .yaml, and mixing the
two up is easy. This renders the map with every cube, pickup pose and delivery
zone marked and labelled, and also prints the numbers.

    python3 tools/show_positions.py                 # writes tools/positions.png
    python3 tools/show_positions.py --world <file>  # a different world variant
    python3 tools/show_positions.py --out /tmp/x.png

READS THE FILE, NOT THE SIMULATION. This tool parses the .world on disk. If you
drag a cube around inside a running Gazebo, the file still holds the old pose and
this tool will happily show you that old pose - use --live to read the scene that
is actually running, and --write-back to persist what you arranged in the GUI:

    python3 tools/show_positions.py --live               # 读仿真并画出来
    python3 tools/show_positions.py --live --write-back  # 把仿真的位置写回世界文件

WHICH POSE WINS: the world file contains the cube models AND a trailing <state>
block saved from an earlier session, so every cube name appears TWICE with
different coordinates. Measured 2026-10-11: the running simulation reports the
MODEL pose (red_cube_1 at -5.525,-5.889), not the state pose (-10,-6.464). This
tool therefore reads the first definition of each name and ignores the state
block, and says so if it finds both.
"""
import argparse
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / 'src/yzbot/mybot_description/worlds/offic_room.world'
MAP_YAML = ROOT / 'src/yzbot/bot_navigation/maps/room_from_world.yaml'
EXECUTOR_CFG = ROOT / 'src/yzbot/mybot/config/fixed_joint_pick_place.yaml'


def read_map(yaml_path):
    """Return (pixels, width, height, resolution, origin_x, origin_y)."""
    text = yaml_path.read_text(encoding='utf-8')
    image = re.search(r'^image:\s*(\S+)', text, re.M).group(1)
    res = float(re.search(r'^resolution:\s*([\d.]+)', text, re.M).group(1))
    origin = re.search(r'^origin:\s*\[\s*([-\d.]+),\s*([-\d.]+)', text, re.M)
    ox, oy = float(origin.group(1)), float(origin.group(2))
    data = (yaml_path.parent / image).read_bytes()
    fields, index = [], 0
    while len(fields) < 4:
        while data[index:index + 1].isspace():
            index += 1
        if data[index:index + 1] == b'#':
            while data[index:index + 1] != b'\n':
                index += 1
            continue
        end = index
        while not data[end:end + 1].isspace():
            end += 1
        fields.append(data[index:end])
        index = end
    index += 1
    width, height = int(fields[1]), int(fields[2])
    return data[index:index + width * height], width, height, res, ox, oy


def split_world(text):
    """Return (outside_state, state_block) - the two halves that both hold poses.

    The world file carries BOTH a `<model>` definition per object AND a trailing
    or embedded `<state world_name=...>` save from an earlier session, so every
    cube name appears twice with different numbers. Measured 2026-10-12: the
    STATE wins - blue_cube_1's model definition said (-8,-3) while the simulation
    reported (-9.995, 0.015), which is the state's LINK pose. Writing only the
    model definition therefore does nothing at all.
    """
    start = text.find("<state ")
    if start < 0:
        return text, ''
    end = text.find("</state>", start)
    end = len(text) if end < 0 else end + len("</state>")
    return text[:start] + text[end:], text[start:end]


def _poses(block, name):
    """Every pose belonging to `name` in `block`: the model pose and the link's."""
    out = []
    for match in re.finditer(r"<model name='" + re.escape(name) + r"'>(.*?)</model>",
                             block, re.S):
        body = match.group(1)
        for pose in re.finditer(r'<pose>\s*([-\d.eE]+)\s+([-\d.eE]+)', body):
            out.append((float(pose.group(1)), float(pose.group(2))))
    return out


def read_cubes(world_path):
    """Return (cubes, duplicates) using the pose the simulation actually obeys.

    Prefers the <state> block's link pose, falling back to the model definition;
    the two disagree in the shipped world, and only one of them is real.
    """
    text = world_path.read_text(encoding='utf-8')
    outside, state = split_world(text)
    names = sorted(set(re.findall(r"<model name='((?:red|blue)_cube_\d)'", text)))
    cubes, duplicates = {}, []
    for name in names:
        poses = _poses(state, name) or _poses(outside, name)
        if not poses:
            continue
        # The link pose is the last one inside a <model>; the model pose is first.
        cubes[name] = poses[-1] if len(poses) > 1 else poses[0]
        if len(_poses(outside, name)) and len(_poses(state, name)):
            duplicates.append(name)
    return cubes, sorted(set(duplicates))


def read_executor_config(path):
    import yaml
    params = yaml.safe_load(path.read_text(encoding='utf-8'))[
        'fixed_joint_pick_place']['ros__parameters']
    pickups, zones = {}, {}
    for key, value in params.items():
        if key.startswith('pickup_') and key.endswith('_pose'):
            pickups[key[len('pickup_'):-len('_pose')]] = (value[0], value[1])
        elif key.startswith('warehouse_') and key.endswith('_pose') \
                and 'approach' not in key:
            zones[key[len('warehouse_'):-len('_pose')].upper()] = (value[0], value[1])
    return pickups, zones


def read_live_cubes(timeout_sec=15.0):
    """Read the cube poses from the RUNNING simulation via /gazebo/link_states."""
    import rclpy
    from gazebo_msgs.msg import LinkStates
    rclpy.init()
    node = rclpy.create_node('show_positions_live')
    live = {}

    def _cb(msg):
        for index, name in enumerate(msg.name):
            match = re.match(r'((?:red|blue)_cube_\d)::link$', name)
            if match:
                live[match.group(1)] = (round(msg.pose[index].position.x, 5),
                                        round(msg.pose[index].position.y, 5))

    node.create_subscription(LinkStates, '/gazebo/link_states', _cb, 10)
    deadline = time.time() + timeout_sec
    while time.time() < deadline and len(live) < 10:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return live


def write_back(world_path, live):
    """Write positions into BOTH the <state> block and the model definitions.

    Only touching the model definition is what silently did nothing before: the
    state block overrides it, and within the state block it is the LINK pose -
    not the model pose - that Gazebo restores.
    """
    text = world_path.read_text(encoding='utf-8')
    outside, state = split_world(text)
    changed = []
    for name, (x, y) in live.items():
        def _patch(block, label):
            pattern = re.compile(
                r"(<model name='" + re.escape(name) + r"'>)(.*?)(</model>)", re.S)

            def _sub(match):
                body = match.group(2)
                # The inertial pose is RELATIVE to the link frame, so writing
                # world coordinates into it moves the centre of mass off the
                # body. Measured 2026-10-12 with gzserver alone (no robot, no
                # ROS, nothing touching them): the ten cubes carrying their
                # world x,y in the inertial pose drifted 0.2644 m in total and
                # rotated up to 2.914 rad in 45 s, while the same world with
                # `0 0 0 0 -0 0` moved by exactly 0.0000 m / 0.000 rad.
                # The <state> link pose overrides the model and link poses, but
                # nothing overrides the inertial pose, so this corruption goes
                # straight into the physics. Shield it from the rewrite.
                shielded = []

                def _shield(m):
                    shielded.append(m.group(0))
                    return f'\x00{len(shielded) - 1}\x00'

                body = re.sub(
                    r'<inertial>.*?</inertial>', _shield, body, flags=re.S)

                def _pose(m):
                    # keep everything after the x y pair (z and orientation)
                    tail = m.group(0)[m.end(2) - m.start(0):]
                    return f'<pose>{x} {y}{tail}'

                body = re.sub(
                    r'<pose>\s*([-\d.eE]+)\s+([-\d.eE]+)([^<]*)</pose>',
                    _pose, body)
                for index, block in enumerate(shielded):
                    body = body.replace(f'\x00{index}\x00', block)
                return match.group(1) + body + match.group(3)

            return pattern.sub(_sub, block, count=1)

        outside = _patch(outside, 'model')
        state = _patch(state, 'state')
        changed.append(name)
    rebuild = text
    if state:
        start = text.find("<state ")
        end = text.find("</state>", start)
        end = len(text) if end < 0 else end + len("</state>")
        rebuild = text[:start] + state + text[end:]
    start = rebuild.find("<state ")
    if start < 0:
        rebuild = outside
    else:
        end = rebuild.find("</state>", start)
        end = len(rebuild) if end < 0 else end + len("</state>")
        rebuild = outside[:start] + rebuild[start:end] + outside[start:]
    world_path.write_text(rebuild, encoding='utf-8')
    return changed


def to_pixel(x, y, width, height, res, ox, oy):
    """World metres -> image pixel. The PGM's first row is the TOP of the map,
    which is the LARGEST y, so the row index has to be flipped."""
    col = int(round((x - ox) / res))
    row = height - 1 - int(round((y - oy) / res))
    return col, row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', default=str(WORLD))
    parser.add_argument('--map', default=str(MAP_YAML))
    parser.add_argument('--executor', default=str(EXECUTOR_CFG))
    parser.add_argument('--out', default=str(Path(__file__).with_name('positions.png')))
    parser.add_argument('--live', action='store_true',
                        help='读运行中的仿真（/gazebo/link_states），而不是世界文件')
    parser.add_argument('--write-back', action='store_true',
                        help='配合 --live：把仿真里的位置写回世界文件')
    parser.add_argument('--suggest-pickups', action='store_true',
                        help='按原有 ±0.58 m 规律算出新的 pickup_*_pose 并打印，可直接粘贴')
    parser.add_argument('--scale', type=int, default=3,
                        help='放大倍数，便于看清标签（默认 3）')
    args = parser.parse_args(argv)

    from PIL import Image, ImageDraw

    pixels, width, height, res, ox, oy = read_map(Path(args.map))
    file_cubes, duplicates = read_cubes(Path(args.world))
    pickups, zones = read_executor_config(Path(args.executor))
    cubes, live = file_cubes, None
    if args.live:
        live = read_live_cubes()
        if not live:
            print('读不到仿真里的物块：仿真没在跑，或 /gazebo/link_states 不可用。',
                  file=sys.stderr)
            return 2
        cubes = live
        print('位置来源: **运行中的仿真**（/gazebo/link_states），不是世界文件\n')
        print(f"{'物块':14s}{'世界文件':>22}{'仿真实时':>22}   差异")
        for name in sorted(set(file_cubes) | set(live)):
            old, now = file_cubes.get(name), live.get(name)
            if old and now:
                gap = math.hypot(old[0] - now[0], old[1] - now[1])
                flag = '一致' if gap < 0.02 else f'差 {gap:.2f} m'
                print(f'  {name:12s}{f"({old[0]:.2f}, {old[1]:.2f})":>22}'
                      f'{f"({now[0]:.2f}, {now[1]:.2f})":>22}   {flag}')
        print()

    image = Image.new('RGB', (width, height))
    image.putdata([(255, 255, 255) if p >= 250 else
                   (40, 40, 40) if p <= 50 else (170, 170, 170)
                   for p in pixels])
    draw = ImageDraw.Draw(image)

    def mark(point, colour, label, radius=6):
        col, row = to_pixel(point[0], point[1], width, height, res, ox, oy)
        if not (0 <= col < width and 0 <= row < height):
            print(f'  ! {label} 在地图范围外: ({point[0]}, {point[1]})', file=sys.stderr)
            return
        draw.ellipse([col - radius, row - radius, col + radius, row + radius],
                     outline=colour, width=3)
        draw.text((col + radius + 2, row - radius), label, fill=colour)

    for name, point in sorted(cubes.items()):
        colour = (200, 30, 30) if name.startswith('red') else (30, 60, 200)
        mark(point, colour, name.replace('_cube_', ''), 7)
    for name, point in sorted(pickups.items()):
        mark(point, (110, 110, 110), '', 3)
    for name, point in sorted(zones.items()):
        mark(point, (0, 140, 0), f'ZONE {name}', 10)

    if args.scale > 1:
        image = image.resize((width * args.scale, height * args.scale),
                             Image.NEAREST)
    image.save(args.out)
    if args.live and args.write_back:
        changed = write_back(Path(args.world), live)
        print(f'已把 {len(changed)} 个物块的位置写回 {args.world}')
        print('别忘了重编: colcon build --packages-select mybot_description mybot '
              '--symlink-install')
    print(f'地图 {width}x{height} px  分辨率 {res} m/px  原点 ({ox}, {oy})')
    print(f'已写出 {args.out}\n')
    print(f"{'物块':16s}{'世界坐标 (x, y)':>24}{'像素 (col, row)':>20}   取物点")
    for name in sorted(cubes):
        x, y = cubes[name]
        col, row = to_pixel(x, y, width, height, res, ox, oy)
        near = pickups.get(name)
        extra = f'({near[0]}, {near[1]})' if near else '-'
        print(f'  {name:14s}{f"({x}, {y})":>24}{f"({col}, {row})":>20}   {extra}')
    print()
    for name, point in sorted(zones.items()):
        col, row = to_pixel(point[0], point[1], width, height, res, ox, oy)
        print(f'  ZONE {name}: ({point[0]}, {point[1]})  ->  像素 ({col}, {row})')
    lethal = {(c, r) for r in range(height) for c in range(width)
              if pixels[r * width + c] < 100}
    print(f"{'物块':16s}{'地图上该点':>12}   最近的目标区（直线距离）")
    blocked = []
    for name in sorted(cubes):
        col, row = to_pixel(*cubes[name], width, height, res, ox, oy)
        if (col, row) in lethal:
            where, blocked = '占用(墙/货架)', blocked + [name]
        else:
            where = '可通行'
        near = sorted((math.hypot(cubes[name][0] - zx, cubes[name][1] - zy), z)
                      for z, (zx, zy) in zones.items())
        print(f'  {name:14s}{where:>12}   '
              + '  '.join(f'{z}={d:.1f}m' for d, z in near))
    if blocked:
        print(f'\n❌ 这些物块落在墙/货架里，机器人抓不到: {", ".join(blocked)}')
    # The trap when moving a cube: the world pose alone is not enough. The
    # executor drives to `pickup_<cube>_pose` from fixed_joint_pick_place.yaml
    # and then searches visually, so a stale pickup pose means it parks at the
    # old spot and never finds the cube.
    stale = []
    for name, (x, y) in sorted(cubes.items()):
        near = pickups.get(name)
        if near is None:
            continue
        gap = math.hypot(x - near[0], y - near[1])
        if gap > 0.8:
            stale.append((name, gap))
    if stale:
        print('\n⚠ 物块挪了、取物点没跟着改（机器人会开去旧位置然后找不到）：')
        for name, gap in stale:
            x, y = cubes[name]
            pick = pickups[name]
            print(f'   {name}: 物块 ({x}, {y})  取物点 ({pick[0]}, {pick[1]})'
                  f'  相差 {gap:.2f} m')
        print('   要一起改: config/fixed_joint_pick_place.yaml 里的 '
              'pickup_<物块名>_pose')
    if args.suggest_pickups:
        standoff = 0.58                      # measured: every pickup is cube ± 0.58 m in x
        print('\n可直接粘贴到 config/fixed_joint_pick_place.yaml 的取物点：')
        print('（沿用原来的 ±0.58 m 站距与朝向；若该侧被墙挡住会自动换到另一侧）')
        for name in sorted(cubes):
            x, y = cubes[name]
            original = pickups.get(name)
            side = 1.0
            if original:
                side = 1.0 if original[0] - file_cubes[name][0] >= 0 else -1.0
            for candidate_side in (side, -side):
                px = x + candidate_side * standoff
                col, row = to_pixel(px, y, width, height, res, ox, oy)
                if 0 <= col < width and 0 <= row < height \
                        and (col, row) not in lethal:
                    yaw = math.pi if candidate_side > 0 else 0.0
                    note = '' if candidate_side == side else '   # 原来那侧被挡住，换了一侧'
                    print(f'    pickup_{name}_pose: [{px:.3f}, {y:.3f}, '
                          f'{yaw:.6f}]{note}')
                    break
            else:
                print(f'    # {name}: 两侧都不可通行，请手动指定')
    if duplicates:
        print(f'\n注意: {", ".join(duplicates)} 在世界文件里出现了多次。')
        print('      本工具只用**第一处**（模型定义）——实测那才是生效的，')
        print('      文件末尾的 <state> 存档不生效。改位置时别改错那一段。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
