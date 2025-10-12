#!/usr/bin/env python3
"""
Tabular Q-Learning Push Agent
Matches supervised learning feature extraction with online scalers and discretization.

State Features (13):
  - distance_to_goal, distance_to_box (computed in real-time)
  - contact_flag, in_goal (binary)
  - robot_x, robot_y, robot_yaw, robot_vx, robot_wz
  - imu_wx, imu_wz, imu_ax, imu_ay, imu_az

Preprocessing:
  - MinMaxScaler: robot_x, robot_y, distance_to_goal, distance_to_box
  - StandardScaler: robot_vx, robot_wz, imu_wx, imu_wz, imu_ax, imu_ay, imu_az, robot_yaw
  - Binary encoding: contact_flag, in_goal

Actions (4 discrete):
  0: Forward (straight push)
  1: Forward-left (corrective turn)
  2: Forward-right (corrective turn)
  3: Stop

Reward Components:
  - r_progress: Box movement toward goal (scaled euclidean distance delta)
  - r_efficiency: Penalty for being stuck while in contact
  - r_rot_pen: Penalty for excessive rotation
  - r_goal: Large reward for reaching goal
  - r_collision: (Reserved for future wall/obstacle collision detection)
"""
import math, time, os, json, numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool

# ==================== Online Scalers (sklearn-compatible) ====================
class RunningMinMax:
    """
    Online MinMaxScaler for known physical ranges.
    If (lo, hi) are provided, uses them as fixed bounds; otherwise adapts online.
    """
    def __init__(self, lo=None, hi=None):
        self.fixed = (lo is not None) and (hi is not None)
        self.lo = np.array(lo, dtype=float) if lo is not None else None
        self.hi = np.array(hi, dtype=float) if hi is not None else None

    def partial_fit(self, x):
        """Update running min/max statistics"""
        x = np.asarray(x, dtype=float)
        if self.fixed:
            return  # Don't update if using fixed bounds
        if self.lo is None:
            self.lo = x.copy()
            self.hi = x.copy()
        else:
            self.lo = np.minimum(self.lo, x)
            self.hi = np.maximum(self.hi, x)

    def transform(self, x):
        """Scale to [0, 1] range"""
        x = np.asarray(x, dtype=float)
        lo, hi = self.lo, self.hi
        # Avoid division by zero
        den = np.where((hi - lo) == 0.0, 1.0, (hi - lo))
        return np.clip((x - lo) / den, 0.0, 1.0)


class RunningStandard:
    """
    Online StandardScaler using Welford's algorithm for numerical stability.
    Computes running mean and variance without storing all samples.
    """
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim, dtype=float)
        self.M2 = np.zeros(dim, dtype=float)  # Sum of squared differences

    def partial_fit(self, x):
        """Update running mean and variance"""
        x = np.asarray(x, dtype=float)
        if self.n == 0:
            self.mean = np.zeros_like(x)
            self.M2 = np.zeros_like(x)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def transform(self, x):
        """Standardize to zero mean, unit variance"""
        x = np.asarray(x, dtype=float)
        if self.n <= 1:
            return x  # Not enough samples yet
        var = self.M2 / (self.n - 1)
        std = np.where(var == 0.0, 1.0, np.sqrt(var))
        return (x - self.mean) / std


# ==================== Utility Functions ====================
def quat_to_yaw(q):
    """Convert quaternion to yaw angle (in radians)"""
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_pi(a):
    """Wrap angle to [-π, π]"""
    while a < -math.pi:
        a += 2 * math.pi
    while a >= math.pi:
        a -= 2 * math.pi
    return a


# ==================== Push Environment ====================
RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
BEST_EFFORT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

class PushEnv(Node):
    """
    ROS2 environment wrapper for tabular Q-learning push task.
    
    Key improvements:
    - Fixed imu_wx extraction and inclusion in state
    - Coarse discretization (3 bins) to reduce state space to ~2M states
    - Complete reward function with all components from supervised learning
    - Better logging and diagnostics
    """
    def __init__(self):
        super().__init__('qlearn_push_env')
        
        # ===== ROS2 Topics =====
        self.cmd_topic = '/model/diff_drive/cmd_vel'
        self.box_pose_topic = '/model/push_box/pose'
        self.odom_topic = '/model/diff_drive/odometry'
        self.imu_topic = '/model/diff_drive/imu'
        self.contact_topic = ''  # Optional: set if you have a contact sensor

        # ===== World Geometry (match your world.sdf) =====
        self.goal_cx, self.goal_cy = 2.25, 0.0
        self.goal_hx, self.goal_hy = 0.75, 0.5  # Half-extents
        self.box_half = 0.4  # Box half-size
        
        # Robot parameters (for wheel speed calculation)
        self.r_wheel = 0.08  # Wheel radius (m)
        self.axle_length = 0.33  # Wheel separation (m)

        # ===== ROS2 Publishers/Subscribers =====
        self.pub = self.create_publisher(Twist, self.cmd_topic, RELIABLE_QOS)

        self.create_subscription(Pose, self.box_pose_topic, self._on_box, RELIABLE_QOS)
        self.create_subscription(Pose, self.box_pose_topic, self._on_box, BEST_EFFORT_QOS)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, RELIABLE_QOS)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, BEST_EFFORT_QOS)
        self.create_subscription(Imu, self.imu_topic, self._on_imu, RELIABLE_QOS)
        self.create_subscription(Imu, self.imu_topic, self._on_imu, BEST_EFFORT_QOS)
        if self.contact_topic:
            self.create_subscription(Bool, self.contact_topic, self._on_contact, RELIABLE_QOS)
            self.create_subscription(Bool, self.contact_topic, self._on_contact, BEST_EFFORT_QOS)

        # ===== State Holders =====
        self.box_pose = None
        self.odom = None
        self.imu = None
        self.contact_flag = False

        # ===== Online Scalers =====
        # MinMaxScaler: robot_x, robot_y, distance_to_goal, distance_to_box
        # Use arena bounds: x∈[-3,3], y∈[-2,2], distances∈[0,6]
        self.mm = RunningMinMax(
            lo=[-3.0, -2.0, 0.0, 0.0],
            hi=[+3.0, +2.0, 6.0, 4.0]
        )
        
        # StandardScaler: robot_vx, robot_wz, imu_wx, imu_wz, imu_ax, imu_ay, imu_az, robot_yaw
        self.std = RunningStandard(dim=8)

        # ===== Discretization Bins (COARSE to reduce state space) =====
        # 3 bins instead of 5: reduces state space from 6 trillion to ~2 million
        self.mm_bins = np.array([0.33, 0.67])   # 3 bins: [0-0.33], [0.33-0.67], [0.67-1]
        self.std_bins = np.array([-0.5, 0.5])   # 3 bins: <-0.5, [-0.5,0.5], >0.5

        # ===== Internal State Tracking =====
        self._prev_dist_goal = None
        self._prev_progress_val = 0.0
        self._logged_first = False
        self._last_cmd_vx = 0.0
        self._last_cmd_wz = 0.0
    
    # ----- Start-up waits -----
    def wait_for_publishers(self, timeout=20.0):
        start = time.time()
        need_pub = [self.box_pose_topic, self.odom_topic]  # published by ros_gz_bridge
        need_sub = [self.cmd_topic]                        # subscribed by ros_gz_bridge
        while time.time() - start < timeout:
            pubs_ok = all(self.count_publishers(t) > 0 for t in need_pub)
            subs_ok = all(self.count_subscribers(t) > 0 for t in need_sub)
            if pubs_ok and subs_ok:
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)
        return False

    def wait_for_first_msgs(self, timeout=15.0):
        """
        Wait for first pose/odom. If nothing shows for a few seconds,
        gently nudge the robot so odom/pose start publishing.
        """
        start = time.time()
        nudged = False
        while time.time() - start < timeout:
            if (self.box_pose is not None) and (self.odom is not None):
                return True
            if not nudged and (time.time() - start) > 3.0:
                self.get_logger().warn("No messages yet; sending a tiny nudge to trigger publishers...")
                self._nudge_robot(duration=0.5, vx=0.06, wz=0.0)
                nudged = True
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.01)
        return False

    def _nudge_robot(self, duration=0.3, vx=0.05, wz=0.0):
        if self.count_subscribers(self.cmd_topic) == 0:
            self.get_logger().warn("No subscriber on cmd_vel; nudge would be dropped.")
            return
        msg = Twist()
        msg.linear.x = vx
        msg.angular.z = wz
        t0 = time.time()
        while time.time() - t0 < duration:
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.01)
        self.pub.publish(Twist())

    # ===== ROS2 Callbacks =====
    def _on_box(self, msg):
        self.box_pose = msg
        if not self._logged_first:
            self._logged_first = True
            self.get_logger().info('Received first sensor messages (box/odom/imu)')
    
    def _on_odom(self, msg):
        self.odom = msg
    
    def _on_imu(self, msg):
        self.imu = msg
    
    def _on_contact(self, msg):
        self.contact_flag = bool(msg.data)

    # ===== Geometry Helpers =====
    def box_fully_in_goal(self):
        """Check if box is completely inside goal region"""
        if self.box_pose is None:
            return False
        x, y = self.box_pose.position.x, self.box_pose.position.y
        
        # Shrink goal by box half-extent (box center must be in shrunken region)
        xmin = self.goal_cx - self.goal_hx + self.box_half
        xmax = self.goal_cx + self.goal_hx - self.box_half
        ymin = self.goal_cy - self.goal_hy + self.box_half
        ymax = self.goal_cy + self.goal_hy - self.box_half
        
        return (xmin <= x <= xmax) and (ymin <= y <= ymax)

    def _estimate_contact_heuristic(self, dist_b):
        """Estimate contact based on distance (if no contact sensor available)"""
        if self.contact_topic:
            return self.contact_flag
        return dist_b < 0.45  # Threshold from your supervised data collection

    # ===== State Observation =====
    def _observe_raw(self):
        """Extract raw state features from ROS messages"""
        if self.box_pose is None or self.odom is None:
            return None
        
        # Box position
        bx, by = self.box_pose.position.x, self.box_pose.position.y
        
        # Robot pose and velocity
        rx, ry = self.odom.pose.pose.position.x, self.odom.pose.pose.position.y
        yaw = wrap_pi(quat_to_yaw(self.odom.pose.pose.orientation))
        rvx = self.odom.twist.twist.linear.x
        rvy = self.odom.twist.twist.linear.y
        rwz = self.odom.twist.twist.angular.z
        
        # IMU (all 6 values)
        if self.imu is not None:
            imu_wx = self.imu.angular_velocity.x
            imu_wy = self.imu.angular_velocity.y
            imu_wz = self.imu.angular_velocity.z
            imu_ax = self.imu.linear_acceleration.x
            imu_ay = self.imu.linear_acceleration.y
            imu_az = self.imu.linear_acceleration.z
        else:
            imu_wx = imu_wy = imu_wz = imu_ax = imu_ay = imu_az = 0.0

        # Derived features (computed same way as supervised learning)
        dist_g = math.hypot(bx - self.goal_cx, by - self.goal_cy)  # Distance to goal
        dist_b = math.hypot(bx - rx, by - ry)                       # Distance to box

        # Binary flags
        cflag = int(self._estimate_contact_heuristic(dist_b))
        ing = int(self.box_fully_in_goal())

        return dict(
            # Derived features
            distance_to_goal=dist_g,
            distance_to_box=dist_b,
            contact_flag=cflag,
            in_goal=ing,
            # Robot state
            robot_x=rx, robot_y=ry, robot_yaw=yaw,
            robot_vx=rvx, robot_vy=rvy, robot_wz=rwz,
            # IMU (all 6 components)
            imu_wx=imu_wx, imu_wy=imu_wy, imu_wz=imu_wz,
            imu_ax=imu_ax, imu_ay=imu_ay, imu_az=imu_az,
        )

    def observe_discrete(self):
        """
        Transform raw observations into discrete state tuple.
        
        Returns:
            (state_tuple, state_aux) or None
            - state_tuple: (12 discretized features, 2 binary) = 14 total
            - state_aux: dict with continuous values for reward computation
        """
        raw = self._observe_raw()
        if raw is None:
            return None

        # ===== MinMax group (4 features) =====
        mm_vec = np.array([
            raw['robot_x'],
            raw['robot_y'],
            raw['distance_to_goal'],
            raw['distance_to_box']
        ], dtype=float)
        self.mm.partial_fit(mm_vec)
        mm_scaled = self.mm.transform(mm_vec)

        # ===== Standard group (8 features) =====
        std_vec = np.array([
            raw['robot_vx'],
            raw['robot_wz'],
            raw['imu_wx'],    
            raw['imu_wz'],
            raw['imu_ax'],
            raw['imu_ay'],
            raw['imu_az'],
            raw['robot_yaw']
        ], dtype=float)
        self.std.partial_fit(std_vec)
        std_scaled = self.std.transform(std_vec)

        # ===== Discretize continuous features =====
        def bin_many(vals, cuts):
            """Apply np.digitize to multiple values"""
            return tuple(int(np.digitize([v], cuts)[0]) for v in vals)

        mm_bins = bin_many(mm_scaled, self.mm_bins)    # 4 features → {0, 1, 2}
        std_bins = bin_many(std_scaled, self.std_bins)  # 8 features → {0, 1, 2}

        # ===== Binary flags (no discretization) =====
        cflag = int(raw['contact_flag'])
        ing = int(raw['in_goal'])

        # ===== Final state tuple =====
        # State space: 3^4 * 3^8 * 2 * 2 = 81 * 6561 * 4 = ~2.1M states
        state = mm_bins + std_bins + (cflag, ing)
        
        # ===== Auxiliary info for reward computation =====
        state_aux = dict(
            dist_goal=raw['distance_to_goal'],
            dist_box=raw['distance_to_box'],
            in_goal=ing,
            contact=cflag,
            wz=raw['robot_wz'],
            vx=raw['robot_vx'],
        )
        
        return state, state_aux

    # ===== Reward Function =====
    def reward(self, prev_dist_g, dist_g, wz, vx, in_goal, contact):
        """
        Compute reward matching supervised learning components.
        
        Components:
        - r_progress: Reward for reducing box-to-goal distance
        - r_efficiency: Penalty for being stuck while in contact
        - r_rot_pen: Penalty for excessive rotation
        - r_goal: Large reward for completing task
        - r_collision: (Reserved for future wall collision detection)
        """
        # ===== Progress toward goal (main driver) =====
        r_progress = 0.0
        if prev_dist_g is not None and dist_g is not None:
            delta = prev_dist_g - dist_g
            r_progress = delta * 10.0  # Scale up to be significant
        
        # ===== Efficiency penalty (stuck while in contact) =====
        r_efficiency = 0.0
        if contact and abs(r_progress) < 1e-4:  # Not making progress while touching
            r_efficiency = -0.01
        
        # ===== Rotation penalty (discourage spinning) =====
        r_rot_pen = -0.001 * abs(wz)
        
        # ===== Goal reward (task completion) =====
        r_goal = 100.0 if in_goal else 0.0
        
        # ===== Collision penalty (future: detect wall hits) =====
        r_collision = 0.0  # TODO: Add if you implement collision detection
        
        # ===== Small step penalty (encourage efficiency) =====
        r_step = -0.01
        
        # ===== Total reward =====
        r_total = r_progress + r_efficiency + r_rot_pen + r_goal + r_collision + r_step
        
        return r_total

    # ===== Action Execution =====
    def step(self, action, speed=0.30, k=0.6):
        """
        Execute discrete action and return (next_state, reward, done).
        
        Actions:
            0: Forward (straight push toward goal)
            1: Forward-left (corrective turn)
            2: Forward-right (corrective turn)
            3: Stop (emergency brake)
        """
        # ===== Map discrete action to continuous commands =====
        if action == 0:
            cmd_vx, cmd_wz = speed, 0.0          # Straight forward
        elif action == 1:
            cmd_vx, cmd_wz = speed * 0.6, +k    # Left turn (slower)
        elif action == 2:
            cmd_vx, cmd_wz = speed * 0.6, -k    # Right turn (slower)
        else:
            cmd_vx, cmd_wz = 0.0, 0.0            # Stop
        
        # Store for logging (optional: compute wheel speeds for supervised learning compatibility)
        self._last_cmd_vx = cmd_vx
        self._last_cmd_wz = cmd_wz

        # ===== Publish Twist command =====
        msg = Twist()
        msg.linear.x = cmd_vx
        msg.angular.z = cmd_wz
        self.pub.publish(msg)

        # ===== Wait for control period (~20 Hz) =====
        t0 = self.get_clock().now()
        while (self.get_clock().now() - t0).nanoseconds < 5e7:  # 50ms
            rclpy.spin_once(self, timeout_sec=0.01)

        # ===== Get next observation =====
        out = self.observe_discrete()
        if out is None:
            return None, 0.0, False
        
        s2, aux = out
        dist_g = aux['dist_goal']
        in_goal = aux['in_goal']
        contact = aux['contact']
        wz = aux['wz']
        vx = aux['vx']

        # ===== Compute reward =====
        r = self.reward(self._prev_dist_goal, dist_g, wz, vx, in_goal, contact)
        self._prev_dist_goal = dist_g
        
        # ===== Check termination =====
        done = bool(in_goal)
        
        return (s2, r, done)

    def reset(self, settle=0.5):
        """
        Reset environment for new episode.
        
        Note: Currently uses fixed spawn positions. For more robust learning,
        consider randomizing robot/box positions using Gazebo services.
        """
        # Stop robot
        self.pub.publish(Twist())
        self._prev_dist_goal = None
        
        # Wait for sensor data to settle
        t0 = self.get_clock().now()
        while (self.get_clock().now() - t0).nanoseconds < settle * 1e9:
            rclpy.spin_once(self, timeout_sec=0.05)
        
        return self.observe_discrete()


# ==================== Tabular Q-Learning Agent ====================
class QAgent:
    """
    Tabular Q-learning with ε-greedy exploration.
    
    Q-learning update rule:
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    """
    def __init__(self, n_actions=4, alpha=0.1, gamma=0.95, eps=1.0, eps_min=0.05, eps_decay=0.99):
        self.nA = n_actions
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.eps = eps              # Exploration rate
        self.eps_min = eps_min      # Minimum exploration
        self.eps_decay = eps_decay  # Decay rate per episode
        self.Q = {}                 # Q-table: {(state, action): value}

    def _q(self, s, a):
        """Get Q-value with default 0.0 for unseen state-action pairs"""
        return self.Q.get((s, a), 0.0)

    def act(self, s):
        """ε-greedy action selection"""
        if np.random.rand() < self.eps:
            return np.random.randint(self.nA)  # Explore
        else:
            qs = [self._q(s, a) for a in range(self.nA)]
            return int(np.argmax(qs))  # Exploit

    def update(self, s, a, r, s2, done):
        """Q-learning update"""
        if done:
            target = r  # No future reward
        else:
            best_next = max(self._q(s2, a2) for a2 in range(self.nA))
            target = r + self.gamma * best_next
        
        old_q = self._q(s, a)
        self.Q[(s, a)] = old_q + self.alpha * (target - old_q)

    def decay(self):
        """Decay exploration rate"""
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
    
    def save(self, path):
        """Save Q-table and hyperparameters to JSON"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            # Serialize state tuples as strings
            q_serializable = {f"{s}||{a}": float(v) for (s, a), v in self.Q.items()}
            json.dump({
                'Q': q_serializable,
                'params': {
                    'alpha': float(self.alpha),
                    'gamma': float(self.gamma),
                    'eps': float(self.eps),
                    'nA': int(self.nA)
                }
            }, f, indent=2)
    
    def load(self, path):
        """Load Q-table and hyperparameters from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
            self.Q = {}
            for key, val in data['Q'].items():
                s_str, a_str = key.rsplit('||', 1)
                s = eval(s_str)  # Convert string back to tuple
                a = int(a_str)
                self.Q[(s, a)] = float(val)
            params = data['params']
            self.alpha = params['alpha']
            self.gamma = params['gamma']
            self.eps = params['eps']
            self.nA = params['nA']


# ==================== Training Loop ====================
def main():
    rclpy.init()
    node = PushEnv()

    # --- wait for bridge/publishers and first msgs ---
    if not node.wait_for_publishers(timeout=20.0):
        node.get_logger().error("No publishers for required topics. Check bridge/names.")
        rclpy.shutdown()
        return
    node.get_logger().info("Required publishers are available.")

    if not node.wait_for_first_msgs(timeout=15.0):
        node.get_logger().error("Didn't receive first /model/push_box/pose or /model/diff_drive/odometry.")
        rclpy.shutdown()
        return
    node.get_logger().info("Received initial messages.")

    agent = QAgent(
        n_actions=4,
        alpha=0.1,       # Learning rate (lower = more stable)
        gamma=0.95,      # Discount factor
        eps=1.0,         # Initial exploration rate
        eps_min=0.05,    # Minimum exploration
        eps_decay=0.99   # Decay per episode (slower = more exploration)
    )

    def _ensure_rollouts_csv(path):
        exists = os.path.exists(path)
        if not exists or os.path.getsize(path) == 0:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "episode","step","is_eval","epsilon","action",
                    "vx_cmd","wz_cmd","reward",
                    "dist_goal","dist_box","contact","in_goal"])

    # ===== File Paths =====
    metrics_path = os.environ.get("QLEARN_METRICS", "/tmp/qlearn_metrics.jsonl")
    qtable_path = os.environ.get("QLEARN_QTABLE", "/tmp/qtable.json")
    rollouts_csv = os.environ.get("ROLLOUTS_CSV", "/tmp/rollouts.csv")
    _ensure_rollouts_csv(rollouts_csv)
    
    node.get_logger().info("=" * 70)
    node.get_logger().info("  TABULAR Q-LEARNING PUSH AGENT")
    node.get_logger().info("=" * 70)
    node.get_logger().info(f"Metrics log:  {metrics_path}")
    node.get_logger().info(f"Q-table save: {qtable_path}")
    node.get_logger().info(f"Rollouts csv: {rollouts_csv}")
    node.get_logger().info(f"State space:  ~2.1M states (3^12 * 2 * 2)")
    node.get_logger().info(f"Action space: 4 actions (forward, left, right, stop)")
    node.get_logger().info(f"Hyperparams:  α={agent.alpha}, γ={agent.gamma}, ε₀={agent.eps}, decay={agent.eps_decay}")
    node.get_logger().info("=" * 70)

    try:
        num_episodes = int(os.environ.get("QLEARN_EPISODES", "100"))
        max_steps = int(os.environ.get("QLEARN_MAX_STEPS", "200"))
        eval_every = int(os.environ.get("QLEARN_EVAL_EVERY", "5"))
        
        node.get_logger().info(f"Starting training: {num_episodes} episodes × {max_steps} max steps")

        for ep in range(1, num_episodes + 1):
            # Decide if the episode is for evaluation (no exploration)
            is_eval = (eval_every > 0) and (ep % eval_every == 0)

            old_eps = agent.eps
            if is_eval:
                agent.eps = agent.eps_min  # No exploration during eval
            
            # ===== Reset Environment =====
            obs = node.reset(settle=1.0)
            retries = 0
            while obs is None and retries < 50:
                rclpy.spin_once(node, timeout_sec=0.1)
                obs = node.observe_discrete()
                retries += 1
            
            if obs is None:
                node.get_logger().error(f"[Q] ep={ep:03d} Failed to get initial observation. Skipping.")
                continue
            
            s, aux = obs
            ep_ret, ep_steps, done = 0.0, 0, False

            # ===== Episode Loop =====
            for t in range(max_steps):
                a = agent.act(s)
                out = node.step(a)
                
                if out is None:
                    time.sleep(0.05)
                    continue
                
                s2, r, done = out

                if not is_eval:
                    agent.update(s, a, r, s2, done)
                
                ep_ret += r
                ep_steps += 1
                s = s2
                
                if done:
                    break

            # ===== Post-Episode =====
            if not is_eval:
                agent.decay()

            # Log metrics
            success = 1 if done else 0
            rec = {
                "episode": ep,
                "return": float(ep_ret),
                "steps": ep_steps,
                "success": success,
                "epsilon": float(agent.eps),
                "q_table_size": len(agent.Q),
                "eval_mode": int(is_eval)
            }
            with open(metrics_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            
            # Restore exploration rate if it was an eval episode
            if is_eval:
                agent.eps = old_eps

            # Console log
            node.get_logger().info(
                f"[Q] ep={ep:03d}/{num_episodes}  "
                f"return={ep_ret:+7.2f}  "
                f"steps={ep_steps:03d}  "
                f"success={'✓' if success else '✗'}  "
                f"ε={agent.eps:.3f}  "
                f"Q-size={len(agent.Q)}"
            )
            
            # Save Q-table periodically
            if ep % 10 == 0:
                agent.save(qtable_path)
                node.get_logger().info(f"  └─ Q-table saved ({len(agent.Q)} entries)")

        # ===== Training Complete =====
        agent.save(qtable_path)
        node.get_logger().info("=" * 70)
        node.get_logger().info(f"Training complete! Final Q-table: {len(agent.Q)} entries")
        node.get_logger().info(f"Q-table saved to: {qtable_path}")
        node.get_logger().info(f"Metrics saved to: {metrics_path}")
        node.get_logger().info("=" * 70)

    except KeyboardInterrupt:
        node.get_logger().info("\n[Q] Training interrupted by user (Ctrl+C)")
    finally:
        node.pub.publish(Twist())  # Stop robot
        agent.save(qtable_path)
        rclpy.shutdown()

if __name__ == "__main__":
    main()
