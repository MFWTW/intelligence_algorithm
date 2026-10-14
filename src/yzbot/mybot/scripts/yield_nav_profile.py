"""Create an opt-in Nav2 profile with the additional YieldPause BT library.

Humble BtNavigator replaces, rather than appends to, plugin_lib_names when a
parameter is supplied. Therefore this list is pinned to Nav2 1.1.20's default
BtNavigator plugin_libs; do not activate the mode on a different Nav2 version
without verifying both the stock launch hash and this plugin list.
"""

import tempfile
from pathlib import Path

import yaml

BT_PLUGINS_1_1_20 = (
    'nav2_compute_path_to_pose_action_bt_node',
    'nav2_compute_path_through_poses_action_bt_node',
    'nav2_compute_and_track_route_bt_node',
    'nav2_compute_route_bt_node',
    'nav2_smooth_path_action_bt_node',
    'nav2_follow_path_action_bt_node',
    'nav2_spin_action_bt_node',
    'nav2_wait_action_bt_node',
    'nav2_assisted_teleop_action_bt_node',
    'nav2_back_up_action_bt_node',
    'nav2_drive_on_heading_bt_node',
    'nav2_clear_costmap_service_bt_node',
    'nav2_is_stuck_condition_bt_node',
    'nav2_goal_reached_condition_bt_node',
    'nav2_initial_pose_received_condition_bt_node',
    'nav2_goal_updated_condition_bt_node',
    'nav2_globally_updated_goal_condition_bt_node',
    'nav2_is_path_valid_condition_bt_node',
    'nav2_reinitialize_global_localization_service_bt_node',
    'nav2_rate_controller_bt_node',
    'nav2_distance_controller_bt_node',
    'nav2_speed_controller_bt_node',
    'nav2_truncate_path_action_bt_node',
    'nav2_truncate_path_local_action_bt_node',
    'nav2_goal_updater_node_bt_node',
    'nav2_recovery_node_bt_node',
    'nav2_pipeline_sequence_bt_node',
    'nav2_round_robin_node_bt_node',
    'nav2_transform_available_condition_bt_node',
    'nav2_time_expired_condition_bt_node',
    'nav2_path_expiring_timer_condition',
    'nav2_distance_traveled_condition_bt_node',
    'nav2_single_trigger_bt_node',
    'nav2_goal_updated_controller_bt_node',
    'nav2_is_battery_low_condition_bt_node',
    'nav2_navigate_through_poses_action_bt_node',
    'nav2_navigate_to_pose_action_bt_node',
    'nav2_remove_passed_goals_action_bt_node',
    'nav2_planner_selector_bt_node',
    'nav2_controller_selector_bt_node',
    'nav2_goal_checker_selector_bt_node',
    'nav2_controller_cancel_bt_node',
    'nav2_path_longer_on_approach_bt_node',
    'nav2_wait_cancel_bt_node',
    'nav2_spin_cancel_bt_node',
    'nav2_assisted_teleop_cancel_bt_node',
    'nav2_back_up_cancel_bt_node',
    'nav2_drive_on_heading_cancel_bt_node',
    'nav2_is_battery_charging_condition_bt_node',
)


def guarded_executor_profile(source):
    """Opt-in executor parameter switch without changing the production file."""
    params = yaml.safe_load(Path(source).read_text(encoding='utf-8'))
    executor = params['fixed_joint_pick_place']['ros__parameters']
    executor['retain_nav_goal_on_yield'] = True
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', suffix='.yaml', prefix='mybot_yield_executor_',
        delete=False,
    ) as output:
        yaml.safe_dump(params, output, sort_keys=False)
        return output.name


def guarded_profile(source):
    """Write a temporary copy; never change a scored-round profile in place."""
    params = yaml.safe_load(Path(source).read_text(encoding='utf-8'))
    navigator = params['bt_navigator']['ros__parameters']
    existing = navigator.get('plugin_lib_names')
    if existing is not None and existing != list(BT_PLUGINS_1_1_20):
        raise RuntimeError('Nav2 plugin list changed; refuse guarded pause mode')
    navigator['plugin_lib_names'] = [*BT_PLUGINS_1_1_20, 'mybot_yield_pause_bt']
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', suffix='.yaml', prefix='mybot_yield_nav_',
        delete=False,
    ) as output:
        yaml.safe_dump(params, output, sort_keys=False)
        return output.name
