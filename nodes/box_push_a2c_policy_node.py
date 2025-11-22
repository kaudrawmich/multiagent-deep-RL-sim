#!/usr/bin/env python3
"""
A2C policy node for box-pushing.

- Subscribes:  observations (std_msgs/Float64MultiArray with 8 dims)
               [dx_rb, dy_rb, dist_bg, heading_err, push_dir_err, v_lin, v_ang, contact]
- Publishes:   /model/diff_drive/cmd_vel (geometry_msgs/Twist)
- Mode:        'train' (sample actions, learn) or 'eval' (greedy, no learn)
- Reward:      progress, push alignment, contact bonus, step cost, |ω| penalty, success bonus
- Update:      On-policy minibatches with GAE(λ); shared torso with policy logits + value head
- Exploration: entropy bonus β decays episode-by-episode → exploitation bias
"""

import math
import numpy as np
import tensorflow as tf
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist

def cosf(x: float) -> float:
    return float(np.cos(x))

@dataclass
class Hyper:
    obs_dim: int = 8
    num_actions: int = 5
    gamma: float = 0.98
    gae_lambda: float = 0.95
    lr: float = 1e-3
    value_coef: float = 0.5
    entropy_start: float = 0.02      # initial β (small; you asked to bias exploitation)
    entropy_end: float = 0.001       # final β
    entropy_anneal_episodes: int = 300
    max_steps: int = 800
    horizon: int = 64                # rollout length before each update
    batch_updates: int = 1           # times to update per rollout
    k1: float = 1.0                  # progress weight
    k2: float = 0.2                  # push alignment weight
    k3: float = 0.2                  # contact+alignment bonus
    k4: float = 0.01                 # step cost
    k5: float = 0.01                 # |ω| penalty
    R_goal: float = 50.0             # success bonus
    goal_radius: float = 0.20

class A2CBoxPush(Node):
    def __init__(self):
        super().__init__('box_push_a2c_policy_node', automatically_declare_parameters_from_overrides=False)

        def param_str(name, default):
            if self.has_parameter(name):
                return self.get_parameter(name).get_parameter_value().string_value
            return self.declare_parameter(name, default).get_parameter_value().string_value

        def param_bool(name, default):
            if self.has_parameter(name):
                return self.get_parameter(name).get_parameter_value().bool_value
            return self.declare_parameter(name, default).get_parameter_value().bool_value

        def param_float(name, default):
            if self.has_parameter(name):
                return float(self.get_parameter(name).get_parameter_value().double_value)
            return float(self.declare_parameter(name, default).get_parameter_value().double_value)

        # Parameters
        self.mode        = param_str('mode', 'train')
        self.cmd_topic   = param_str('cmd_topic', '/model/diff_drive/cmd_vel')
        self.goal_radius = param_float('goal_radius', 0.20)
        self.use_sim_time= param_bool('use_sim_time', True)

        # Hyperparameters
        self.h = Hyper(goal_radius=self.goal_radius)

        # ROS I/O
        self.obs = np.zeros((self.h.obs_dim,), dtype=np.float32)
        self.have_obs = False
        self.create_subscription(Float64MultiArray, 'observations', self._on_obs, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

        # NN
        self.model = self._build_model(self.h.obs_dim, self.h.num_actions)
        self.opt = tf.keras.optimizers.Nadam(self.h.lr)

        # Rollout buffers
        self.reset_episode()
        self.roll_S = []
        self.roll_A = []
        self.roll_R = []
        self.roll_V = []
        self.roll_logp = []
        self.roll_done = []

        # Timer (20 Hz)
        self.timer = self.create_timer(0.05, self._tick)
        self.get_logger().info(f"A2C up in {self.mode} mode; cmd_topic={self.cmd_topic}")

    # ---------------- NN ----------------
    def _build_model(self, obs_dim: int, num_actions: int) -> tf.keras.Model:
        inp = tf.keras.layers.Input(shape=(obs_dim,), name='obs')
        x = tf.keras.layers.Dense(128, activation='elu')(inp)
        x = tf.keras.layers.Dense(128, activation='elu')(x)
        x = tf.keras.layers.Dense(64, activation='elu')(x)
        logits = tf.keras.layers.Dense(num_actions, name='pi_logits')(x)
        value  = tf.keras.layers.Dense(1, name='v')(x)
        return tf.keras.Model(inputs=inp, outputs=[logits, value])

    # ---------- ROS Callbacks -----------
    def _on_obs(self, msg: Float64MultiArray):
        data = np.asarray(msg.data, dtype=np.float32)
        if data.shape[0] != self.h.obs_dim:
            self.get_logger().warn(f"Bad obs size {data.shape[0]} != {self.h.obs_dim}")
            return
        self.obs = data
        self.have_obs = True

    # --------- Episode / Control --------
    def reset_episode(self):
        self.episode = getattr(self, 'episode', 0) + 1
        self.step = 0
        self.done = False
        self.trunc = False
        self.prev_s = None
        self.prev_action = None
        self.prev_cmd = Twist()
        self.entropy_coef = self._entropy_coef(self.episode)

    def _entropy_coef(self, ep: int) -> float:
        # Linear anneal β from start→end over entropy_anneal_episodes
        h = self.h
        t = min(max(ep, 1), h.entropy_anneal_episodes)
        frac = 1.0 - (t - 1) / max(1, h.entropy_anneal_episodes - 1)
        return h.entropy_end + (h.entropy_start - h.entropy_end) * frac

    def _select_action(self, s: np.ndarray, greedy: bool = False):
        s2d = s.reshape(1, -1)
        logits, v = self.model.predict(s2d, verbose=0)
        logits = logits[0]
        v = float(v[0, 0])

        # Softmax
        probs = tf.nn.softmax(logits).numpy()
        if greedy or self.mode == 'eval':
            a = int(np.argmax(probs))
        else:
            a = int(np.random.choice(self.h.num_actions, p=probs))
        logp_a = float(np.log(max(1e-8, probs[a])))
        return a, logp_a, v

    def _twist_for(self, a: int) -> Twist:
        t = Twist()
        if a == 0:                # Stop
            pass
        elif a == 1:              # Forward (small)
            t.linear.x = 0.20
        elif a == 2:              # Forward (bigger)
            t.linear.x = 0.35
        elif a == 3:              # Turn left in place
            t.angular.z = +0.8
        elif a == 4:              # Turn right in place
            t.angular.z = -0.8
        return t

    # --------- Reward Shaping ----------
    def _reward(self, s_prev: np.ndarray, s_now: np.ndarray) -> float:
        # s = [dx_rb, dy_rb, dist_bg, heading_err, push_dir_err, v_lin, v_ang, contact]
        h = self.h
        r = 0.0
        if s_prev is not None:
            # progress: distance reduction box→goal
            r += h.k1 * (float(s_prev[2]) - float(s_now[2]))
        # pushing towards goal?
        r += h.k2 * cosf(float(s_now[4]))
        # contact + alignment
        if (float(s_now[7]) > 0.5) and (cosf(float(s_now[4])) > 0.5):
            r += h.k3
        # step penalty and |ω| penalty
        r -= h.k4
        r -= h.k5 * abs(float(s_now[6]))
        # success bonus
        if float(s_now[2]) < h.goal_radius:
            r += h.R_goal
            self.done = True
        return float(r)

    # --------- GAE & Update ------------
    def _compute_gae(self, rewards, values, dones, last_v):
        """
        rewards, values, dones: lists length T
        last_v: scalar bootstrap value for V(S_T)
        returns advantages (T,), returns (T,)
        """
        gamma, lam = self.h.gamma, self.h.gae_lambda
        T = len(rewards)
        adv = np.zeros((T,), dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(dones[t])
            next_v = last_v if t == T-1 else values[t+1]
            delta = rewards[t] + gamma * next_v * nonterminal - values[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        rets = adv + np.asarray(values, dtype=np.float32)
        return adv, rets

    def _update_on_rollout(self):
        if len(self.roll_S) < self.horizon:
            return

        S   = np.asarray(self.roll_S,   dtype=np.float32)
        A   = np.asarray(self.roll_A,   dtype=np.int32)
        R   = np.asarray(self.roll_R,   dtype=np.float32)
        V   = np.asarray(self.roll_V,   dtype=np.float32)
        D   = np.asarray(self.roll_done, dtype=np.float32)
        # bootstrap value for last state
        last_v = 0.0 if (len(self.roll_S) == 0) else float(self.model.predict(self.obs.reshape(1,-1), verbose=0)[1][0,0])

        adv, rets = self._compute_gae(R, V, D, last_v)
        # normalize advantages (common trick)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # one update pass (you can increase batch_updates if you want extra epochs)
        for _ in range(self.h.batch_updates):
            with tf.GradientTape() as tape:
                logits, v_pred = self.model(S, training=True)  # [N, A], [N, 1]
                v_pred = tf.squeeze(v_pred, axis=1)            # [N]
                log_probs = tf.nn.log_softmax(logits)          # [N, A]
                entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(log_probs) * log_probs, axis=1))

                a_onehot = tf.one_hot(A, self.h.num_actions)
                logp_a = tf.reduce_sum(a_onehot * log_probs, axis=1)  # [N]
                policy_loss = -tf.reduce_mean(logp_a * tf.constant(adv, dtype=tf.float32))
                value_loss  = tf.reduce_mean(tf.square(v_pred - tf.constant(rets, dtype=tf.float32)))
                loss = policy_loss + self.h.value_coef * value_loss - self.entropy_coef * entropy

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        # clear rollout
        self.roll_S.clear(); self.roll_A.clear(); self.roll_R.clear()
        self.roll_V.clear(); self.roll_logp.clear(); self.roll_done.clear()

    # ------------- Main Loop ------------
    def _tick(self):
        if not self.have_obs:
            return

        self.step += 1
        s = self.obs.copy()

        # Select and publish action
        greedy = (self.mode == 'eval')
        a, logp_a, v_s = self._select_action(s, greedy=greedy)
        cmd = self._twist_for(a)
        self.cmd_pub.publish(cmd)

        # Observe next state in next tick; compute reward using prev/now pair
        r = self._reward(self.prev_s, s)
        self.prev_s = s

        # episode truncate
        if self.step >= self.h.max_steps:
            self.trunc = True

        # Store transition (on-policy)
        if self.mode == 'train':
            self.roll_S.append(s)
            self.roll_A.append(a)
            self.roll_R.append(r)
            self.roll_V.append(v_s)
            self.roll_logp.append(logp_a)
            self.roll_done.append(float(self.done or self.trunc))
            self._update_on_rollout()

        # End of episode?
        if self.done or self.trunc:
            # Small stop cmd
            self.cmd_pub.publish(Twist())

            # Log & reset
            self.get_logger().info(
                f"Episode {self.episode} end: steps={self.step} "
                f"done={self.done} trunc={self.trunc} β={self.entropy_coef:.4f}"
            )
            # clear any partial rollout to avoid mixing episodes
            self.roll_S.clear(); self.roll_A.clear(); self.roll_R.clear()
            self.roll_V.clear(); self.roll_logp.clear(); self.roll_done.clear()
            self.reset_episode()

def main():
    rclpy.init()
    node = A2CBoxPush()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
