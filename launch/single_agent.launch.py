#launch/single_agent.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, TextSubstitution, LaunchConfiguration
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    namePackage = 'multiagent_deep_rl_sim'
    robotXacroName = 'diff_drive'

    model_rel = 'models/diff_drive/robot.xacro'
    world_rel = 'worlds/robot_arena.world'

    pkg_share = get_package_share_directory(namePackage)
    pathModelFile = os.path.join(pkg_share, model_rel)
    pathWorldFile = os.path.join(pkg_share, world_rel)

    # Start the controller
    start_controller_arg = DeclareLaunchArgument('start_controller', default_value='false')
    start_obs_arg = DeclareLaunchArgument('start_obs', default_value='true')
    start_rl_arg = DeclareLaunchArgument('start_rl', default_value='true')

    # xacro → URDF
    robotDescription = xacro.process_file(pathModelFile).toxml()

    # Gazebo environment variables
    set_gz = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[
            EnvironmentVariable('GZ_SIM_RESOURCE_PATH', default_value=''),
            TextSubstitution(text=os.pathsep),
            pkg_share,
            TextSubstitution(text=os.pathsep),
            os.path.join(pkg_share, 'models'),
        ],
    )
    set_ign = SetEnvironmentVariable(  # compatibility
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=[
            EnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', default_value=''),
            TextSubstitution(text=os.pathsep),
            pkg_share,
            TextSubstitution(text=os.pathsep),
            os.path.join(pkg_share, 'models'),
            TextSubstitution(text=os.pathsep),
            os.path.join(pkg_share, 'worlds'),
        ],
    )

    # Include ros_gz_sim’s gazebo launcher
    gz_launch_src = PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
    )
    gazeboLaunch = IncludeLaunchDescription(
        gz_launch_src,
        launch_arguments={
            'gz_args': f'-r -v 4 {pathWorldFile}',
            'on_exit_shutdown': 'true',
        }.items()
    )

    # Spawn robot from /robot_description
    spawnModelNodeGazebo = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', robotXacroName, 
                   '-topic', 'robot_description',
                   '-x', '-2.7',
                   '-y', '0.0',
                   '-z', '0.00',
                   '-R', '0.0',
                   '-P', '0.0',
                   '-Y', '0.0'],
        output='screen',
    )

    # Spawn box from its SDF file
    box_sdf = os.path.join(pkg_share, 'models', 'push_box', 'model.sdf')
    spawnBoxNodeGazebo = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', 'push_box',
                   '-file', box_sdf,
                   '-x', '-2.0',
                   '-y', '0.0',
                   '-z', '0.40',
                   '-R', '0.0',
                   '-P', '0.0',
                   '-Y', '0.0'],
        output='screen',
    )

    # Publish robot state
    nodeRobotStatePublisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robotDescription, 'use_sim_time': True}],
    )

    # Bridge config
    bridge_params = os.path.join(pkg_share, 'bridge', 'robot_bridge.yaml')
    start_gazebo_ros_bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_params}'],
        output='screen',
    )

    controller = Node(
        package=namePackage,
        executable='push_to_goal.py',
        name='push_to_goal',
        output='screen',
        parameters=[{'use_sim_time': True, 'topic': '/model/diff_drive/cmd_vel', 'creep_speed': 0.0}],
        condition=IfCondition(LaunchConfiguration('start_controller')),
    )

    # Observation Node
    obs = Node(
        package=namePackage,
        executable='box_push_observation_node.py',
        name='box_push_observation',
        output='screen',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('start_obs')),
    )

    # Policy Node
    rl = Node(
        package=namePackage,
        executable='box_push_a2c_policy_node.py',
        name='box_push_a2c_policy_node',
        output='screen',
        parameters=[{'mode': 'train',
                     'cmd_topic': '/model/diff_drive/cmd_vel',
                     'goal_radius': 0.20}],
        condition=IfCondition(LaunchConfiguration('start_rl')),
    )

    ld = LaunchDescription()
    for a in (start_controller_arg, start_obs_arg, start_rl_arg):
        ld.add_action(a)
    
    ld.add_action(set_gz)
    ld.add_action(set_ign)
    ld.add_action(gazeboLaunch)
    ld.add_action(spawnModelNodeGazebo)
    ld.add_action(spawnBoxNodeGazebo)
    ld.add_action(nodeRobotStatePublisher)
    ld.add_action(start_gazebo_ros_bridge_cmd)

    # Optional controller
    ld.add_action(controller)

    # RL nodes and observation
    ld.add_action(obs)
    ld.add_action(rl)
    return ld
