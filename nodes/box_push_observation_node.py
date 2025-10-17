#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray

def yaw_from_quat(q):
    # q: geometry_msgs/Quaternion
    siny_cosp = 2.0*(q.w*q.z + q.x*q.y)
    cosy_cosp = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def wrap(a):
    while a >  math.pi: a -= 2.0*math.pi
    while a < -math.pi: a += 2.0*math.pi
    return a

class BoxPushObservationNode(Node):
    def __init__(self):
        super().__init__('box_push_observation_node', automatically_declare_parameters_from_overrides=True)

        # State holders
        self.robot_pose = (0.0, 0.0, 0.0)   # x,y,yaw
        self.robot_twist = (0.0, 0.0)       # v_lin, v_ang
        self.box_xy  = (0.0, 0.0)
        self.goal_xy = (2.25, 0.0)          # default; will be updated by topic if bridged
        self.contact = 0.0                  # reserved for later (contact sensor)
        self.have_robot = False
        self.have_box   = False

        self.create_subscription(Odometry, '/model/diff_drive/odometry', self._on_odom, 10)
        self.create_subscription(Pose, '/model/push_box/pose',           self._on_box_pose, 10)
        self.create_subscription(Pose, '/model/goal_region/pose',        self._on_goal_pose, 10)

        self.pub = self.create_publisher(Float64MultiArray, 'observations', 10)
        self.timer = self.create_timer(0.05, self._tick)  # 20 Hz

    def _on_odom(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        v_lin = msg.twist.twist.linear.x
        v_ang = msg.twist.twist.angular.z
        self.robot_pose = (x,y,yaw)
        self.robot_twist = (v_lin, v_ang)
        if not self.have_robot:
            self.get_logger().info("Received first robot odom")
        self.have_robot = True

    def _on_box_pose(self, msg: Pose):
        self.box_xy = (msg.position.x, msg.position.y)
        if not self.have_box:
            self.get_logger().info("Received first box pose")
        self.have_box = True

    def _on_goal_pose(self, msg: Pose):
        self.goal_xy = (msg.position.x, msg.position.y)
        if not self.have_goal:
            self.get_logger().info("Received first goal pose")
        self.have_goal = True

    def _tick(self):
        if not (self.have_robot and self.have_box and self.have_goal):
            return

        rx, ry, ryaw = self.robot_pose
        bx, by = self.box_xy
        gx, gy = self.goal_xy
        v_lin, v_ang = self.robot_twist

        dx_rb = bx - rx
        dy_rb = by - ry
        dist_bg = math.hypot(gx - bx, gy - by)

        heading_to_box = math.atan2(dy_rb, dx_rb)
        heading_err    = wrap(heading_to_box - ryaw)

        push_dir       = math.atan2(gy - by, gx - bx)       # goal-from-box
        robot_to_box   = math.atan2(by - ry, bx - rx)       # box-from-robot
        push_dir_err   = wrap(push_dir - robot_to_box)

        obs = [dx_rb, dy_rb, dist_bg, heading_err, push_dir_err, v_lin, v_ang, self.contact]
        msg = Float64MultiArray(); msg.data = obs
        self.pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(BoxPushObservationNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
