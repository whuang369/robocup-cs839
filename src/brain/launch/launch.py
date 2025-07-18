import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def handle_configuration(context, *args, **kwargs):
    config_path = os.path.join(os.path.dirname(__file__), "../config")
    robot = context.perform_substitution(LaunchConfiguration("robot"))
    sim = context.perform_substitution(LaunchConfiguration("sim"))

    config = {}
    config_file = os.path.join(config_path, f"brain_{robot}.yaml")
    if sim in ["true", "True", "1"]:
        config["use_sim_time"] = True
        config_file = os.path.join(config_path, "brain_sim.yaml")

    # (Optional) Local configuration file for debugging
    config_local_file = os.path.join(config_path, "brain_local.yaml")

    behavior_trees_dir = os.path.join(os.path.dirname(__file__), "../behavior_trees")

    def make_tree_path(name):
        if not name.endswith(".xml"):
            name += ".xml"
        return os.path.join(behavior_trees_dir, name)

    # override parameters from launch arguments
    tree = context.perform_substitution(LaunchConfiguration("tree"))
    config["tree_file_path"] = make_tree_path(tree)

    role = context.perform_substitution(LaunchConfiguration("role"))
    if not role == "":
        config["game.player_role"] = role
    attack_side = context.perform_substitution(LaunchConfiguration("attack"))
    if not attack_side == "":
        config["game.player_attack_side"] = attack_side
    start_pos = context.perform_substitution(LaunchConfiguration("pos"))
    if not start_pos == "":
        config["game.player_start_pos"] = float(start_pos)
    rerun = context.perform_substitution(LaunchConfiguration("rerun"))
    if not rerun == "":
        config["rerunLog.server_addr"] = f"rerun+http://{rerun}/proxy"

    return [
        Node(
            package="brain",
            executable="brain_node",
            name="brain_node",
            output="screen",
            parameters=[config_file, config_local_file, config],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot",
                default_value="r1",
                description="Robot name (selects brain_<robot>.yaml)",
            ),
            DeclareLaunchArgument(
                "sim",
                default_value="false",
                description="Run in simulation mode (selects brain_sim.yaml if true)",
            ),
            DeclareLaunchArgument(
                "tree",
                default_value="game.xml",
                description="Specify behavior tree file name. DO NOT include full path, file should be in src/brain/config/behavior_trees",
            ),
            DeclareLaunchArgument(
                "role",
                default_value="",
                description="If you need to override the game.player_role in the config.yaml, you can specify the parameter role:=striker when launching",
            ),
            DeclareLaunchArgument(
                "attack",
                default_value="",
                description="If you need to override the game.player_attack_side in the config.yaml"
            ),
            DeclareLaunchArgument(
                "pos",
                default_value="",
                description="If you need to override the game.player_start_pos in the config.yaml, you can specify the parameter pos:=left when launching.",
            ),
            DeclareLaunchArgument(
                "rerun", default_value="", description="Override the rerunLog server address"
            ),
            OpaqueFunction(function=handle_configuration),
        ]
    )
