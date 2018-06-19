from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import time
import numpy as np

from gym import spaces
from gym.utils import seeding
from gym_brt.qube.envs.QubeServo2 import QubeServo2


# theta, alpha: positions, velocities, accelerations
OBSERVATION_HIGH = np.asarray([
    1, 1, 1, 1, # angles
    np.pi / 4, np.pi / 4, # velocities
    np.pi / 4, np.pi / 4, # accelerations
    4100, # tach0
    0.2, # sense
], dtype=np.float64)
OBSERVATION_LOW = -OBSERVATION_HIGH

MAX_MOTOR_VOLTAGE = 8.0
ACTION_HIGH = np.asarray([MAX_MOTOR_VOLTAGE], dtype=np.float64)
ACTION_LOW = -ACTION_HIGH

WARMUP_STEPS = 100


STATE_KEYS = [ 
        'COS_THETA',
        'SIN_THETA',
        'COS_ALPHA',
        'SIN_ALPHA',
        'THETA_VELOCITY',
        'ALPHA_VELOCITY',
        'THETA_ACCELERATION',
        'ALPHA_ACCELERATION',
        'TACH0',
        'SENSE'
        ]


def normalize_angle(theta):
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


class QubeInvertedPendulumReward(object):

    def __init__(self):
        self.target_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH, dtype=np.float32) 

    def __call__(self, state, action):
        theta_x = state[0]
        theta_y = state[1]
        alpha_x = state[2]
        alpha_y = state[3]
        theta_velocity = state[4]
        alpha_velocity = state[5]
        theta_acceleration = state[6]
        alpha_acceleration = state[7]

        theta = np.arctan2(theta_y, theta_x) # arm
        alpha = np.arctan2(alpha_y, alpha_x) # pole

        cost =  normalize_angle(theta)**4 + \
                normalize_angle(alpha)**2 + \
                0.1 * alpha_velocity**2

        reward = -cost
        return reward


class QubeInvertedPendulumEnv(gym.Env):

    def __init__(self):
        self.observation_space = spaces.Box(
                OBSERVATION_LOW, OBSERVATION_HIGH, 
                dtype=np.float32)

        self.action_space = spaces.Box(
                ACTION_LOW, ACTION_HIGH, 
                dtype=np.float32)

        self.reward_fn = QubeInvertedPendulumReward()

        self._theta = 0
        self._alpha = 0
        self._theta_velocity = 0
        self._alpha_velocity = 0
        self._theta_acceleration = 0
        self._alpha_acceleration = 0
        self._motor_voltage = 0
        self._tach0 = 0
        self._sense = 0
        self._prev_t = time.time()

        # Open the Qube
        self.qube = QubeServo2(frequency=25)
        self.qube.__enter__()

        self.seed()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        M_PI = 3.14159

        motor_voltages = np.clip(np.array([action[0]], dtype=np.float64), ACTION_LOW, ACTION_HIGH)
        currents, encoders, others = self.qube.action(motor_voltages)

        encoder0 = encoders[0]
        encoder1 = encoders[1] % 2048
        if (encoder1 < 0):
            encoder1 = 2048 + encoder1

        theta_next = encoder0 * (-2.0 * M_PI / 2048);
        alpha_next = encoder1 * (2.0 * M_PI / 2048) - M_PI
        self._sense = currents[0]
        self._tach0 = others[0]

        theta_velocity_next = normalize_angle(theta_next - self._theta)
        alpha_velocity_next = normalize_angle(alpha_next - self._alpha)

        self._theta_acceleration = normalize_angle(theta_velocity_next - self._theta_velocity)
        self._alpha_acceleration = normalize_angle(alpha_velocity_next - self._alpha_velocity)

        self._theta_velocity = theta_velocity_next
        self._alpha_velocity = alpha_velocity_next

        self._theta = theta_next
        self._alpha = alpha_next

        return self._get_state()

    def _get_state(self):
        state = np.asarray([
            np.cos(self._theta),
            np.sin(self._theta),
            np.cos(self._alpha),
            np.sin(self._alpha),
            self._theta_velocity,
            self._alpha_velocity,
            self._theta_acceleration,
            self._alpha_acceleration,
            self._tach0,
            self._sense,
        ], dtype=np.float32)
        return state

    def reset(self):
        # step once with no action to init state
        if WARMUP_STEPS > 0:
            for i in range(WARMUP_STEPS):
                action = np.zeros(
                    shape=self.action_space.shape,
                    dtype=self.action_space.dtype)
                return self._step(action)
        else:
            action = np.zeros(
                shape=self.action_space.shape,
                dtype=self.action_space.dtype)
            return self._step(action)

    def step(self, action):
        state = self._step(action)
        reward = self.reward_fn(state, action)
        done = False
        info = {}
        return state, reward, done, info

    def close(self):
        # Safely close the Qube
        self.qube.__exit__(None, None, None)


def main():
    num_episodes = 10
    num_steps = 250

    with QubeEnv() as env:
        for episode in range(num_episodes):
            state = env.reset()
            for step in range(num_steps):
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                if done:
                    break
                state = next_state

    """
    # Another way to run the Qube enviroment
    env = QubeEnv()

    num_episodes = 10
    num_steps = 250
    try:
        for episode in range(num_episodes):
            state = env.reset()
            for step in range(num_steps):
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                if done:
                    break
                state = next_state
    finally:
        # Note: to set all encoders and motor voltages to 0, you must call env.close()
        env.close()
    """

if __name__ == '__main__':
    main()
