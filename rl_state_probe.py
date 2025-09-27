#!/usr/bin/env python3
import math
import threading
import time
from dataclasses import dataclass, asdict

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from ros_gz_interfaces.msg import Contacts


def quat_to_yaw(x, y, z, w):
    # ZYX yaw from quaternion
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


@dataclass
class Pose2D:
    x: float = float("nan")
    y: float = float("nan")
    yaw: float = float("nan")


class ArenaProbe(Node):
    def __init__(self):
        super().__init__('arena_probe')

        # --- params you can tweak quickly
        self.declare_parameter('goal_size_x', 1.5)   # goal width in X (m)
        self.declare_parameter('goal_size_y', 1.0)   # goal height in Y (m)
        self.declare_parameter('print_hz', 5.0)      # summary print rate
        self.declare_parameter('auto_nudge', False)  # simple test motion
        self.declare_parameter('nudge_linear', 0.25) # m/s
        self.declare_parameter('nudge_angular', 0.0) # rad/s
        self.declare_parameter('nudge_duration', 2.0)# seconds

        # --- publishers
        self.cmd_pub = self.create_publisher(Twist, '/model/robot_1/cmd_vel', 10)

        # --- subscribers
        self.create_subscription(Odometry, '/model/robot_1/odometry', self.cb_odom, 10)
        self.create_subscription(Imu, '/model/robot_1/imu', self.cb_imu, 10)
        self.create_subscription(Contacts, '/model/robot_1/pusher_contact', self.cb_contacts, 10)

        self.create_subscription(Pose, '/model/robot_1/pose', self.cb_robot_pose, 10)
        self.create_subscription(Pose, '/model/push_box/pose', self.cb_box_pose, 10)
        self.create_subscription(Pose, '/model/goal_region/pose', self.cb_goal_pose, 10)
        self.create_subscription(Pose, '/model/obstacle_upper/pose', self.cb_obs1_pose, 10)
        self.create_subscription(Pose, '/model/obstacle_lower/pose', self.cb_obs2_pose, 10)

        # --- state containers
        self.robot = Pose2D()
        self.box   = Pose2D()
        self.goal  = Pose2D()
        self.obs1  = Pose2D()
        self.obs2  = Pose2D()
        self.yaw_rate = float("nan")
        self.contact_flag = 0

        # --- periodic printing
        period = 1.0 / float(self.get_parameter('print_hz').value)
        self.create_timer(period, self.print_summary)

        # --- optional simple motion (for a quick push test)
        if bool(self.get_parameter('auto_nudge').value):
            threading.Thread(target=self._nudge_thread, daemon=True).start()

        self.get_logger().info("ArenaProbe node started.")

    # ======== Callbacks ========
    def cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot.x = p.x
        self.robot.y = p.y
        self.robot.yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

    def cb_robot_pose(self, msg: Pose):
        # (Optional) if you prefer pose over odom for position
        p = msg.position
        q = msg.orientation
        self.robot.x = p.x
        self.robot.y = p.y
        self.robot.yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

    def cb_box_pose(self, msg: Pose):
        p = msg.position
        q = msg.orientation
        self.box.x = p.x
        self.box.y = p.y
        self.box.yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

    def cb_goal_pose(self, msg: Pose):
        p = msg.position
        q = msg.orientation
        self.goal.x = p.x
        self.goal.y = p.y
        self.goal.yaw = quat_to_yaw(q.x, q.y, q.z, q.w)  # should be ~0

    def cb_obs1_pose(self, msg: Pose):
        p = msg.position
        self.obs1.x = p.x
        self.obs1.y = p.y
        self.obs1.yaw = 0.0

    def cb_obs2_pose(self, msg: Pose):
        p = msg.position
        self.obs2.x = p.x
        self.obs2.y = p.y
        self.obs2.yaw = 0.0

    def cb_imu(self, msg: Imu):
        # Yaw rate is angular velocity around Z (ENU)
        self.yaw_rate = msg.angular_velocity.z

    def cb_contacts(self, msg: Contacts):
        self.contact_flag = 1 if len(msg.contacts) > 0 else 0

    # ======== Computations ========
    def _vec_and_dist(self, a: Pose2D, b: Pose2D):
        dx = b.x - a.x
        dy = b.y - a.y
        return dx, dy, math.hypot(dx, dy)

    def _box_inside_goal(self):
        # Goal is axis-aligned in world (your world uses 0 yaw), size from params
        gx = float(self.get_parameter('goal_size_x').value)
        gy = float(self.get_parameter('goal_size_y').value)
        dx = self.box.x - self.goal.x
        dy = self.box.y - self.goal.y
        return (abs(dx) <= gx * 0.5) and (abs(dy) <= gy * 0.5)

    def rl_observation(self):
        # Build a simple list (fine for a first RL test)
        (rb_dx, rb_dy, rb_dist) = self._vec_and_dist(self.robot, self.goal)
        (bx_dx, bx_dy, bx_dist) = self._vec_and_dist(self.box, self.goal)

        obs = {
            "robot_pose": asdict(self.robot),
            "imu_yaw_rate": self.yaw_rate,
            "contact": int(self.contact_flag),
            "box_pose": asdict(self.box),
            "goal_pose": asdict(self.goal),
            "goal_vec_robot": {"dx": rb_dx, "dy": rb_dy, "dist": rb_dist},
            "goal_vec_box":   {"dx": bx_dx, "dy": bx_dy, "dist": bx_dist},
            "box_in_goal": bool(self._box_inside_goal()),
            "obstacles": [
                asdict(self.obs1),
                asdict(self.obs2),
            ],
        }
        return obs

    # ======== Output / demo control ========
    def print_summary(self):
        obs = self.rl_observation()
        # Only print once we have the essential values (not NaN)
        if not math.isnan(self.robot.x) and not math.isnan(self.box.x) and not math.isnan(self.goal.x):
            self.get_logger().info(
                f"robot(xy,yaw)=({self.robot.x:+.2f},{self.robot.y:+.2f},{self.robot.yaw:+.2f}) | "
                f"box(xy,yaw)=({self.box.x:+.2f},{self.box.y:+.2f},{self.box.yaw:+.2f}) | "
                f"goalΔ(box)=({obs['goal_vec_box']['dx']:+.2f},{obs['goal_vec_box']['dy']:+.2f}) "
                f"d={obs['goal_vec_box']['dist']:.2f} m | "
                f"imu_yaw_rate={self.yaw_rate:+.3f} | contact={self.contact_flag} | "
                f"in_goal={obs['box_in_goal']}"
            )

    def _nudge_thread(self):
        time.sleep(1.0)  # give Gazebo + bridge a moment
        v = float(self.get_parameter('nudge_linear').value)
        w = float(self.get_parameter('nudge_angular').value)
        dur = float(self.get_parameter('nudge_duration').value)

        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w

        t_end = time.time() + dur
        self.get_logger().info(f"Auto-nudge: publishing v={v:.2f} m/s, w={w:.2f} rad/s for {dur:.1f}s")
        rate = 50.0
        dt = 1.0 / rate
        while time.time() < t_end and rclpy.ok():
            self.cmd_pub.publish(msg)
            time.sleep(dt)

        # stop
        self.cmd_pub.publish(Twist())
        self.get_logger().info("Auto-nudge: stop.")
        

def main():
    rclpy.init()
    node = ArenaProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
