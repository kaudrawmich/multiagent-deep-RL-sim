#!/usr/bin/env python3
import os
import math
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool

def quat_to_yaw(q):
    """
    Convert a quaternion into a yaw angle (in radians)
    Assumes the quaternion is normalized

    Args:
        q: geometry_msgs.msg.Quaternion
    
    Returns:
        yaw angle in radians
    """
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

class PushToGoal(Node):
    """
    Simple controller to push a box to a goal region
    """
    def __init__(self):
        super().__init__('push_to_goal')

        # --- Parameters ---
        # CLI Overrides
        self.declare_parameter('topic', '/model/diff_drive/cmd_vel')
        self.declare_parameter('speed', 0.30)
        self.declare_parameter('seconds', 0.0)
        self.declare_parameter('k_yaw', 0.6)
        self.declare_parameter('log_file', '')
        self.declare_parameter('creep_speed', 0.15)
        self.declare_parameter('exit_on_done', True)
        self.declare_parameter('episode_id', 1)

        # Topic names
        self.declare_parameter('box_pose_topic', '/model/push_box/pose')
        self.declare_parameter('odom_topic', '/model/diff_drive/odometry')
        self.declare_parameter('imu_topic', '/model/diff_drive/imu')
        self.declare_parameter('contact_flag_topic', '')

        # Goal region parameters
        self.declare_parameter('goal_center_x', 2.25)
        self.declare_parameter('goal_center_y', 0.00)
        self.declare_parameter('goal_half_x', 1.5 / 2.0) # 0.75
        self.declare_parameter('goal_half_y', 1.0 / 2.0) # 0.5

        # Box parameters
        self.declare_parameter('box_half', 0.8 / 2.0) # 0.4

        # Read parameters
        self.cmd_topic = self.get_parameter('topic').value
        self.vx_cmd = float(self.get_parameter('speed').value)
        self.stop_after = float(self.get_parameter('seconds').value)
        self.k_yaw = float(self.get_parameter('k_yaw').value)
        self.creep_speed = float(self.get_parameter('creep_speed').value)
        self.exit_on_done = self.get_parameter('exit_on_done').value
        self.episode_id = int(self.get_parameter('episode_id').value)
        
        self.box_pose_topic = self.get_parameter('box_pose_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.imu_topic = self.get_parameter('imu_topic').value
        self.contact_topic = self.get_parameter('contact_flag_topic').value

        self.r_wheel = float(self.get_parameter('wheel_radius').value)
        self.axle_length = float(self.get_parameter('axle_length').value)

        self.goal_center_x = float(self.get_parameter('goal_center_x').value)
        self.goal_center_y = float(self.get_parameter('goal_center_y').value)
        self.goal_half_x = float(self.get_parameter('goal_half_x').value)
        self.goal_half_y = float(self.get_parameter('goal_half_y').value)
        self.box_half = float(self.get_parameter('box_half').value)
        
        # Open log file (CSV) if provided
        self.log_fh = None
        log_path = self.get_parameter('log_file').value
        if log_path:
            os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
            self.get_logger().info(f'Logging to {log_path}')
            self.log_fh = open(log_path, 'w')
            header = (
                'episode,step,done,done_reason,'
                'robot_x,robot_y,robot_yaw,robot_vx,robot_vy,robot_wz,'
                'imu_wx,imu_wy,imu_wz,imu_ax,imu_ay,imu_az,'
                'box_x,box_y,box_yaw,'
                'goal_x,goal_y,'
                'contact_flag,'
                'cmd_vx,cmd_wz,wheel_l_rad_s,wheel_r_rad_s,'
                'r_progress,r_collision,r_efficiency,r_rot_pen,r_goal,r_total,'
                'in_goal\n'
            )
            self.log_fh.write(header)
        
        # IO Topics
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.create_subscription(Pose, self.box_pose_topic, self.on_box_pose, 10)
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)
        self.create_subscription(Imu, self.imu_topic, self.on_imu, 10)
        if self.contact_topic:
            self.create_subscription(Bool, self.contact_topic, self.on_contact, 10)

        # State holders
        self.box_pose = None
        self.robot_odom = None
        self.imu_msg = None
        self.contact_flag = False

        # Timers
        self.timer = self.create_timer(0.05, self.control_step) # Control loop @ 20 Hz
        self.print_timer = self.create_timer(1.0, self.print_status) # Print status @ 1 Hz

        # Episode state
        self.reached = False
        self.start_time = self.get_clock().now()
        self.step_idx = 0
        self.prev_dist_box_goal = None
        self.done_reason = 'running'

        # Diagnostics
        self._diag_timer = self.create_timer(1.0, self._diag_once)
        self._logged_wait = False

        # Command echo
        self.last_cmd_vx = 0.0
        self.last_cmd_wz = 0.0
    
    def _diag_once(self):
        """
        Print out publishers for key topics once at startup
        and then cancel this timer

        Returns:
            None
        """
        for t in [self.box_pose_topic, self.odom_topic, self.imu_topic]:
            pubs = self.get_publishers_info_by_topic(t)
            names = [f'{p.node_name}({p.node_namespace})' for p in pubs]
            self.get_logger().info(f'Topic {t} publishers: {names or "NONE"}')
        self._diag_timer.cancel()

    def on_box_pose(self, msg: Pose):
        """
        Box pose callback

        Args:
            msg: geometry_msgs.msg.Pose
        
        Returns: 
            None
        """
        self.box_pose = msg

    def on_odom(self, msg: Odometry):
        """
        Robot odometry callback

        Args:
            msg: nav_msgs.msg.Odometry
        
        Returns:
            None
        """
        self.robot_odom = msg

    def on_imu(self, msg: Imu):
        """
        IMU callback

        Args:
            msg: sensor_msgs.msg.Imu
        
        Returns:
            None
        """
        self.imu_msg = msg
    
    def on_contact(self, msg: Bool):
        """
        Contact sensor callback

        Args:
            msg: std_msgs.msg.Bool
        
        Returns:
            None
        """
        self.contact_flag = bool(msg.data)

    def box_fully_in_goal(self):
        """
        Check if the box is fully inside the goal region

        Returns:
            bool: True if box is fully inside goal, False otherwise
        """
        if self.box_pose is None:
            return False
        
        x = self.box_pose.position.x
        y = self.box_pose.position.y

        # To be "fully inside", the box center must be inside the goal
        # shrunk by the box half-extent on each side.
        xmin = self.goal_center_x - self.goal_half_x + self.box_half
        xmax = self.goal_center_x + self.goal_half_x - self.box_half
        ymin = self.goal_center_y - self.goal_half_y + self.box_half
        ymax = self.goal_center_y + self.goal_half_y - self.box_half
        return (xmin <= x <= xmax) and (ymin <= y <= ymax)
    
    def control_step(self):
        """
        Main control loop step

        Returns:
            None
        """
        if self.reached:
            return
        
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if self.stop_after > 0.0 and elapsed >= self.stop_after:
            self.done_reason = 'timeout'
            self._finish_episode()
            return
        
        # Wait for messages to arrive
        if self.cmd_pub.get_subscription_count() == 0 and not self._logged_wait:
            self._logged_wait = True
            self.get_logger().warn(f'No subscribers to {self.cmd_topic} yet. Waiting...')

        if self.box_pose is None or self.robot_odom is None:
            # Start collecting data once we have the box and odom
            self.send_cmd(self.creep_speed, 0.0)
            self._log_row()
            return
        
        # Termination conditions
        if self.box_fully_in_goal():
            self.reached = True
            self.done_reason = 'goal_reached'
            self._send_cmd(0.0, 0.0)
            self._log_row(done=True)
            if self.exit_on_done:
                self.create_timer(0.01, self._shutdown)
            return
        
        # Simple controller to push box to goal
        by = self.box_pose.position.y
        ang = -self.k_yaw * by
        self.send_cmd(self.vx_cmd, ang)

        # Log row
        self._log_row()

    def send_cmd(self, vx, wz):
        """
        Send velocity command to robot

        Args:
            vx: Linear velocity in m/s
            wz: Angular velocity in rad/s
        
        Returns:
            None
        """
        self.last_cmd_vx = float(vx)
        self.last_cmd_wz = float(wz)
        msg = Twist()
        msg.linear.x = self.last_cmd_vx
        msg.angular.z = self.last_cmd_wz
        self.cmd_pub.publish(msg)
    
    def _finish_episode(self):
        """
        Finish the episode by stopping the robot and logging final state

        Returns:
            None
        """
        self.reached = True
        self._send_cmd(0.0, 0.0)
        self._log_row(done=True)
        if self.exit_on_done:
            self.create_timer(0.01, self._shutdown)
    
    def _shutdown(self):
        """
        Shutdown the node

        Returns:
            None
        """
        self.get_logger().info('Shutting down node')
        rclpy.shutdown()
    
    # Logging helpers
    def _estimate_contact_heuristic(self, rx, ry, bx, by):
        """
        Estimate contact heuristic based on distance between robot and box

        Args:
            rx: Robot x position
            ry: Robot y position
            bx: Box x position
            by: Box y position
        
        Returns:
            bool: True if contact is likely, False otherwise
        """
        if self.contact_topic:
            return self.contact_flag
        if rx is None or ry is None or bx is None or by is None:
            return False
        dist = math.hypot(bx - rx, by - ry)
        return dist < 0.45
    
    def _compute_wheel_speeds(self, vx, wz):
        """
        Compute left and right wheel speeds from linear and angular velocities

        Args:
            vx: Linear velocity in m/s
            wz: Angular velocity in rad/s
        
        Returns:
            (wl, wr): Tuple of left and right wheel speeds in rad/s
        """
        r = self.r_wheel
        L = self.axle_length
        wl = (vx / r) - 0.5 * (wz * L / r)
        wr = (vx / r) + 0.5 * (wz * L / r)
        return wl, wr
    
    def _log_row(self, done=False):
        """
        Log a row of data to the CSV log file if logging is enabled

        Args:
            done: bool indicating if the episode is done
        
        Returns:
            None
        """
        self.step_idx += 1

        # Robot pose/vel
        rx = ry = ryaw = rvx = rvy = rwz = None
        if self.robot_odom:
            rx = self.robot_odom.pose.pose.position.x
            ry = self.robot_odom.pose.pose.position.y
            ryaw = quat_to_yaw(self.robot_odom.pose.pose.orientation)
            rvx = self.robot_odom.twist.twist.linear.x
            rvy = self.robot_odom.twist.twist.linear.y
            rwz = self.robot_odom.twist.twist.angular.z
        
        # IMU
        imu_wx = imu_wy = imu_wz = imu_ax = imu_ay = imu_az = None
        if self.imu_msg:
            imu_wx = self.imu_msg.angular_velocity.x
            imu_wy = self.imu_msg.angular_velocity.y
            imu_wz = self.imu_msg.angular_velocity.z
            imu_ax = self.imu_msg.linear_acceleration.x
            imu_ay = self.imu_msg.linear_acceleration.y
            imu_az = self.imu_msg.linear_acceleration.z
        
        # Box
        bx = by = byaw = None
        if self.box_pose:
            bx = self.box_pose.position.x
            by = self.box_pose.position.y
            byaw = quat_to_yaw(self.box_pose.orientation)
        
        # Goal
        gx = self.goal_center_x
        gy = self.goal_center_y

        # Contact flag (True/False)
        cflag = self._estimate_contact_heuristic(rx, ry, bx, by)

        # Rewards
        r_progress = 0.0 # Decrease in distance from box -> goal
        dist_box_goal = None
        if bx is not None and by is not None:
            dist_box_goal = math.hypot(bx - gx, by - gy)
            if self.prev_dist_box_goal is not None:
                r_progress = self.prev_dist_box_goal - dist_box_goal
            self.prev_dist_box_goal = dist_box_goal
        # Collision penalty
        r_collision = 0.0
        # Efficiency penalty (for using high speeds)
        r_efficiency = -0.01 if (cflag and abs(r_progress < 1e-4)) else 0.0
        # Rotation penalty (for turning in place)
        r_rot_pen = -0.001 * abs(self.last_cmd_wz)
        # Goal rewards
        in_goal = self.box_fully_in_goal()
        r_goal = 10.0 if in_goal else 0.0

        # Total reward
        r_total = r_progress + r_collision + r_efficiency + r_rot_pen + r_goal

        # Done flags / reasons
        done_reason = self.done_reason if done or self.reached else 'running'
        done_str = 'True' if (done or self.reached) else 'False'
        in_goal_str = 'True' if in_goal else 'False'

        # Compute wheel speeds
        wl, wr = self._compute_wheel_speeds(self.last_cmd_vx, self.last_cmd_wz)

        # Write CSV row
        def f(x, default=0.0):
            """
            Format float or use default if None

            Args:
                x: float or None
                default: default value if x is None
            
            Returns:
                float
            """
            return float(x) if x is not None else float(default)
        
        row = [
            int(self.episode_id),
            int(self.step_idx),
            done_str,
            done_reason,
            f(rx), f(ry), f(ryaw),
            f(rvx), f(rvy), f(rwz),
            f(imu_wx), f(imu_wy), f(imu_wz), f(imu_ax), f(imu_ay), f(imu_az),
            f(bx), f(by), f(byaw),
            float(gx), float(gy),
            'True' if cflag else 'False',
            float(self.last_cmd_vx), float(self.last_cmd_wz), float(wl), float(wr),
            float(r_progress), float(r_collision), float(r_efficiency), float(r_rot_pen), float(r_goal), float(r_total),
            in_goal_str
        ]

        if self.log_fh:
            self.log_fh.write(','.join([str(v) for v in row]) + '\n')
            self.log_fh.flush()

    def print_status(self):
        """
        Print current status to console

        Returns:
            None
        """
        if self.box_pose:
            bx = self.box_pose.position.x
            by = self.box_pose.position.y
            self.get_logger().info(f'Box pose: x={bx:.3f}, y={by:.3f}')
        if self.robot_odom:
            rx = self.robot_odom.pose.pose.position.x
            ry = self.robot_odom.pose.pose.position.y
            self.get_logger().info(f'Robot odom: x={rx:.3f}, y={ry:.3f}')

def main():
    rclpy.init()
    node = PushToGoal()
    node.get_logger().info(
        f"push_to_goal up. cmd_topic={node.cmd_topic}, speed={node.vx_cmd:.2f}, "
        f"stop_after={node.stop_after:.1f}s"
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except rclpy.executors.ExternalShutdownException:
        pass
    finally:
        if getattr(node, 'log_fh', None):
            node.log_fh.close()
        node.destroy_node()

if __name__ == '__main__':
    main()
