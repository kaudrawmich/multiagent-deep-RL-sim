#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import math
import time

class PushToGoal(Node):
    def __init__(self):
        super().__init__('push_to_goal')

        # --- Parameters (match your world.sdf) ---
        # CLI Overrides
        self.declare_parameter('topic', '/model/diff_drive/cmd_vel')
        self.declare_parameter('speed', 0.30)
        self.declare_parameter('seconds', 0.0)

        self.cmd_topic = self.get_parameter('topic').value
        self.vx_cmd = float(self.get_parameter('speed').value)
        self.stop_after = float(self.get_parameter('seconds').value)

        # Tweakable parameters
        self.k_yaw = 0.6               # steer toward y=0 with box y
        self.creep_speed = 0.15          # m/s when waiting for box pose

        # Goal region pose and half-size
        self.goal_center_x = 2.15
        self.goal_center_y = 0.0
        self.goal_half_x   = 1.5 / 2.0  # 0.75
        self.goal_half_y   = 1.0 / 2.0  # 0.5

        # Box half-size
        self.box_half = 0.8 / 2.0       # 0.4

        # Simple driving gains
        self.vx_cmd = 0.30              # m/s forward
        self.k_yaw  = 0.6               # steer toward y=0 with box y

        # --- IO ---
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

        self.box_pose = None
        self.robot_odom = None
        self.imu_msg = None
        self._logged_wait = False

        self.create_subscription(Pose, '/model/push_box/pose', self.on_box_pose, 10)
        self.create_subscription(Odometry, '/model/diff_drive/odometry', self.on_odom, 10)
        self.create_subscription(Imu, '/model/diff_drive/imu', self.on_imu, 10)

        # Control loop @ 20 Hz
        self.timer = self.create_timer(0.05, self.control_step)
        # Print status @ 1 Hz
        self.print_timer = self.create_timer(1.0, self.print_status)

        self.reached = False
        self.start_time = time.time()

    # --- Callbacks ---
    def on_box_pose(self, msg: Pose):
        self.box_pose = msg

    def on_odom(self, msg: Odometry):
        self.robot_odom = msg

    def on_imu(self, msg: Imu):
        self.imu_msg = msg

    # --- Geometry check: box fully inside goal ---
    def box_fully_in_goal(self):
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
        if self.reached:
            return
        
        if self.stop_after > 0.0 and (time.time() - self.start_time) >= self.stop_after:
            self.get_logger().info(f'Stopping after {self.stop_after} seconds.')
            self.send_cmd(0.0, 0.0)
            self.reached = True
            return
        
        if self.cmd_pub.get_subscription_count() == 0 and not self._logged_wait:
            self._logged_wait = True
            self.get_logger().warn(f'No subscribers to {self.cmd_topic} yet. Waiting...')

        if self.box_pose is None:
            # Wait until we see the box pose
            self.send_cmd(self.creep_speed, 0.0)
            return

        if self.box_fully_in_goal():
            self.reached = True
            self.get_logger().info('Box is fully inside goal region. Stopping.')
            self.send_cmd(0.0, 0.0)
            return

        # Simple push-forward policy:
        #  - drive forward along +x
        #  - steer toward the arena centerline based on box y
        ang = -self.k_yaw * self.box_pose.position.y
        self.send_cmd(self.vx_cmd, ang)

    def send_cmd(self, vx, wz):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def print_status(self):
        # Periodic prints to screen
        if self.box_pose:
            bx = self.box_pose.position.x
            by = self.box_pose.position.y
            self.get_logger().info(f'Box pose: x={bx:.3f}, y={by:.3f}')
        if self.robot_odom:
            rx = self.robot_odom.pose.pose.position.x
            ry = self.robot_odom.pose.pose.position.y
            self.get_logger().info(f'Robot odom: x={rx:.3f}, y={ry:.3f}')
        if self.imu_msg:
            wz = self.imu_msg.angular_velocity.z
            ax = self.imu_msg.linear_acceleration.x
            self.get_logger().info(f'IMU: wz={wz:.3f} rad/s, ax={ax:.3f} m/s^2')

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
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
