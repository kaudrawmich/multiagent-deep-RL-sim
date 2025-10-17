#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist

def cosf(x): return float(np.cos(x))

class BoxPushDQN(Node):
    def __init__(self):
        super().__init__('box_push_dqn_policy_node', automatically_declare_parameters_from_overrides=True)

        # Obs
        self.obs = np.zeros(8, dtype=np.float32)
        self.create_subscription(Float64MultiArray, 'observations', self._on_obs, 10)
        self.cmd_pub = self.create_publisher(Twist, '/model/diff_drive/cmd_vel', 10)

        # DQN hyperparams
        self.num_actions = 5
        self.gamma = 0.98
        self.epsilon_min = 0.01
        self.max_episodes = 1000
        self.batch_size = 64
        self.learn_start = 200
        self.replay = deque(maxlen=8000)

        self.model = self._build_model()
        self.opt = tf.keras.optimizers.Nadam(1e-3)
        self.loss_fn = tf.keras.losses.MSE

        # Episode bookkeeping
        self.episode = 1
        self.step = 0
        self.max_steps = 800
        self.prev_s = None
        self.done = False
        self.trunc = False

        # Reward shaping constants
        self.k1 = 1.0   # progress (box->goal)
        self.k2 = 0.2   # push alignment
        self.k3 = 0.2   # contact alignment bonus
        self.k4 = 0.01  # step cost
        self.k5 = 0.01  # angular velocity penalty
        self.R_goal = 50.0
        self.goal_radius = 0.20

        self.timer = self.create_timer(0.05, self._tick)  # 20 Hz

    def _build_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Input((8,)),
            tf.keras.layers.Dense(64, activation='elu'),
            tf.keras.layers.Dense(64, activation='elu'),
            tf.keras.layers.Dense(64, activation='elu'),
            tf.keras.layers.Dense(self.num_actions),
        ])

    def _on_obs(self, msg: Float64MultiArray):
        self.obs = np.array(msg.data, dtype=np.float32)

    def _epsilon(self):
        return max(1.0 - self.episode / self.max_episodes, self.epsilon_min)

    def _select_action(self, s):
        if np.random.rand() < self._epsilon():
            return int(np.random.randint(self.num_actions))
        q = self.model.predict(s[np.newaxis], verbose=0)[0]
        return int(np.argmax(q))

    def _twist_for(self, a: int) -> Twist:
        t = Twist()
        if a == 0:             # stop
            pass
        elif a == 1: t.linear.x = 0.20
        elif a == 2: t.linear.x = 0.35
        elif a == 3: t.angular.z = +0.8
        elif a == 4: t.angular.z = -0.8
        return t

    def _reward(self, s_prev, s_now):
        # s = [dx_rb, dy_rb, dist_bg, heading_err, push_dir_err, v_lin, v_ang, contact]
        r = 0.0
        if s_prev is not None:
            r += self.k1 * (s_prev[2] - s_now[2])   # progress: box closer to goal
        r += self.k2 * cosf(s_now[4])               # pushing toward goal?
        if (s_now[7] > 0.5) and (cosf(s_now[4]) > 0.5):
            r += self.k3
        r -= self.k4
        r -= self.k5 * abs(float(s_now[6]))

        # success detection
        if s_now[2] < self.goal_radius:
            r += self.R_goal
            self.done = True
        return float(r)

    def _learn(self):
        if len(self.replay) < max(self.batch_size, self.learn_start): return
        idx = np.random.randint(len(self.replay), size=self.batch_size)
        batch = [self.replay[i] for i in idx]
        S, A, R, S2, D, T = [np.vstack(x) for x in zip(*batch)]

        next_q = self.model.predict(S2, verbose=0)
        max_next = np.max(next_q, axis=1, keepdims=True)
        run_mask = 1.0 - np.clip(D + T, 0, 1)  # stop bootstrapping if done or truncated
        target = R + run_mask * self.gamma * max_next

        with tf.GradientTape() as tape:
            q_all = self.model(S)
            # one-hot gather
            a_onehot = tf.one_hot(A.squeeze().astype(np.int32), self.num_actions)
            q_sa = tf.reduce_sum(q_all * a_onehot, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target, q_sa))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def _reset_episode_bookkeeping(self):
        self.step = 0
        self.done = False
        self.trunc = False
        self.prev_s = None

    def _tick(self):
        # Simple episodic loop without world teleports (you can add a reset service later)
        if self.done or self.trunc or self.step == 0:
            if self.step > 0:
                self.episode += 1
            self._reset_episode_bookkeeping()
            return

        s = self.obs.copy()
        a = self._select_action(s)
        self.cmd_pub.publish(self._twist_for(a))

        self.step += 1
        s2 = self.obs.copy()
        r = self._reward(self.prev_s, s2)
        self.prev_s = s2

        if self.step >= self.max_steps:
            self.trunc = True

        self.replay.append((
            s.reshape(1,-1),
            np.array([[a]]),
            np.array([[r]], dtype=np.float32),
            s2.reshape(1,-1),
            np.array([[float(self.done)]]),
            np.array([[float(self.trunc)]])
        ))

        self._learn()

def main():
    rclpy.init()
    rclpy.spin(BoxPushDQN())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
