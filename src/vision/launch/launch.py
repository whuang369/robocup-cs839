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
    config_file = os.path.join(config_path, f"vision_{robot}.yaml")
    if sim in ["true", "True", "1"]:
        config["use_sim_time"] = True
        config_file = os.path.join(config_path, "vision_sim.yaml")

    # (Optional) Local configuration file for debugging
    config_local_file = os.path.join(config_path, "vision_local.yaml")

    # override parameters from launch arguments
    rerun = context.perform_substitution(LaunchConfiguration("rerun"))
    if not rerun == "":
        config["rerunLog.server_addr"] = f"rerun+http://{rerun}/proxy"
        config["rerunLog.enable"] = True

    return [
        Node(
            package="vision",
            executable="vision_node",
            name="vision_node",
            output="screen",
            arguments=[config_file, config_local_file],
            parameters=[config],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot",
                default_value="r1",
                description="Robot name (selects vision_<robot>.yaml)",
            ),
            DeclareLaunchArgument(
                "sim",
                default_value="false",
                description="Run in simulation mode (selects vision_sim.yaml if true)",
            ),
            DeclareLaunchArgument(
                "rerun",
                default_value="rerun+http://127.0.0.1:9876/proxy",
                description="Override the rerunLog server address",
            ),
            OpaqueFunction(function=handle_configuration),
        ]
    )
