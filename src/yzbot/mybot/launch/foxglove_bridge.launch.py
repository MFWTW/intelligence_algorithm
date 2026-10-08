"""Start the Foxglove WebSocket bridge for the competition visualization.

Standalone use (simulation already running in another terminal)::

    ros2 launch mybot foxglove_bridge.launch.py

``competition_bringup.launch.py`` includes this same file behind its
``foxglove:=`` argument, so the one-click competition start also serves the
dashboard.

Why these parameters:

* ``use_sim_time`` stays ``false`` on purpose.  ``/competition/*`` are
  header-less ``std_msgs`` (String/Bool/Int32), so Foxglove stamps them when
  they arrive over the socket.  With sim time the whole state/progress panel
  set would jump to 0 whenever Gazebo's ``/clock`` is paused or restarted.
* ``topic_whitelist`` defaults to "everything".  The bridge only subscribes to
  topics a connected client actually asks for, so a wide whitelist costs
  nothing; pass ``topic_whitelist:="['/competition/.*','/scan','/tf']"`` to
  trim the Foxglove topic list for the judges.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    port = LaunchConfiguration('port')
    address = LaunchConfiguration('address')
    use_sim_time = LaunchConfiguration('use_sim_time')
    topic_whitelist = LaunchConfiguration('topic_whitelist')
    send_buffer_limit = LaunchConfiguration('send_buffer_limit')

    bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[{
            'port': ParameterValue(port, value_type=int),
            'address': ParameterValue(address, value_type=str),
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            # YAML rules at launch time, so "['a','b']" really arrives as a
            # string array and not as one literal string.
            'topic_whitelist': ParameterValue(topic_whitelist),
            'service_whitelist': ['.*'],
            # 2026-10-06: with param_whitelist ['.*'] plus the parameters /
            # parametersSubscribe / services capabilities the bridge polled every
            # node's parameters and produced 127 "Failed to retrieve parameters"
            # timeouts in one 15-minute round (each one a blocking service call
            # that stalls the bridge's event loop). The competition dashboard is
            # topic-only, so keep the parameter surface off and clip it to the
            # /competition namespace in case a client asks anyway.
            'param_whitelist': ['/competition/.*'],
            'client_topic_whitelist': ['.*'],
            'capabilities': [
                'clientPublish',
                'connectionGraph',
                'assets',
            ],
            'min_qos_depth': 1,
            'max_qos_depth': 10,
            # 10 MB is the upstream default and is quickly exhausted by a
            # 30 Hz camera image when the client stalls; 50 MB keeps the image
            # panel smooth on a local connection without unbounded growth.
            'send_buffer_limit': ParameterValue(send_buffer_limit, value_type=int),
            # sysinfo publishes a 1 Hz host-metrics channel nobody uses here.
            'sysinfo': False,
            'message_backlog_size': 1024,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'port', default_value='8765',
            description='WebSocket port. Foxglove desktop connects to '
                        'ws://localhost:<port>.',
        ),
        DeclareLaunchArgument(
            'address', default_value='0.0.0.0',
            description='Listen address. Keep 0.0.0.0 so the Windows-side '
                        'Foxglove app can reach the WSL container.',
        ),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false', choices=['true', 'false'],
            description='Keep false: /competition/* are header-less and must '
                        'be stamped on arrival, not on Gazebo sim time.',
        ),
        DeclareLaunchArgument(
            'topic_whitelist', default_value="['.*']",
            description='YAML list of regexes (full match) the bridge may '
                        'expose. Narrow it to trim the topic list, e.g. '
                        "'[/competition/.*, /scan, /plan, /tf.*, /camera/.*, /vision/.*, /rosout]'.",
        ),
        DeclareLaunchArgument(
            'send_buffer_limit', default_value='50000000',
            description='Per-connection outbound buffer in bytes.',
        ),
        bridge,
    ])
