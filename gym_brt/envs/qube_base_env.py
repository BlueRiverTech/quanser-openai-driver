from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import time
import math
import numpy as np

from gym import spaces
from gym.utils import seeding
from gym_brt.quanser import QubeServo2
from gym_brt.control import QubeFlipUpControl


# theta, alpha: positions, velocities, accelerations
OBSERVATION_HIGH = np.asarray([
    1, 1, 1, 1,  # angles
    np.pi / 4, np.pi / 4,  # velocities
    np.pi / 4, np.pi / 4,  # accelerations
    4100,  # tach0
    0.2,  # sense
], dtype=np.float64)
OBSERVATION_LOW = -OBSERVATION_HIGH


MAX_MOTOR_VOLTAGE = 8.0
ACTION_HIGH = np.asarray([MAX_MOTOR_VOLTAGE], dtype=np.float64)
ACTION_LOW = -ACTION_HIGH


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


class QubeBaseReward(object):
    def __init__(self):
        self.target_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH, dtype=np.float32)

    def __call__(self, state, action):
        raise NotImplementedError


class QubeBaseEnv(gym.Env):
    def __init__(self, frequency=1000):
        self.observation_space = spaces.Box(
            OBSERVATION_LOW, OBSERVATION_HIGH,
            dtype=np.float32)
        self.action_space = spaces.Box(
            ACTION_LOW, ACTION_HIGH,
            dtype=np.float32)
        self.reward_fn = QubeBaseReward()

        self._theta_velocity_cstate = 0
        self._alpha_velocity_cstate = 0
        self._theta_velocity = 0
        self._alpha_velocity = 0
        self._frequency = frequency

        # Open the Qube
        self.qube = QubeServo2(frequency=frequency)
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
        motor_voltages = np.clip(np.array(
            [action[0]], dtype=np.float64), ACTION_LOW, ACTION_HIGH)
        currents, encoders, others = self.qube.action(motor_voltages)

        self._sense = currents[0]
        self._tach0 = others[0]

        # Calculate alpha, theta, alpha_velocity, and theta_velocity
        self._theta = encoders[0] * (-2.0 * np.pi / 2048)
        alpha_un = encoders[1] * (2.0 * np.pi / 2048)  # Alpha without normalizing
        self._alpha = (alpha_un % (2.0 * np.pi)) - np.pi  # Normalized and shifted alpha

        theta_velocity = -2500 * self._theta_velocity_cstate + 50 * self._theta
        alpha_velocity = -2500 * self._alpha_velocity_cstate + 50 * alpha_un
        self._theta_velocity_cstate += (-50 * self._theta_velocity_cstate + self._theta) / self._frequency
        self._alpha_velocity_cstate += (-50 * self._alpha_velocity_cstate + alpha_un) / self._frequency

        # TODO: update using the transfer function
        self._theta_acceleration = (theta_velocity - self._theta_velocity) * self._frequency
        self._alpha_acceleration = (alpha_velocity - self._alpha_velocity) * self._frequency

        self._theta_velocity = theta_velocity
        self._alpha_velocity = alpha_velocity

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

    def _flip_up(self):
        """Run classic control for flip-up until the pendulum is inverted for
        a set amount of time. Assumes that initial state is stationary
        downwards.
        """
        control = QubeFlipUpControl(env=self, sample_freq=self._frequency)
        time_hold = 1.0 * self._frequency # Number of samples to hold upright
        sample = 0 # Samples since control system started
        samples_upright = 0 # Consecutive samples pendulum is upright

        action = self.action_space.sample()
        state, _, _, _ = self.step([1.0])
        while True:
            action = control.action(state)
            state, _, _, _ = self.step(action)
            # Break if pendulum is inverted
            if self._alpha < (10 * np.pi / 180):
                if samples_upright > time_hold:
                    break
                samples_upright += 1
            else:
                samples_upright = 0
            sample += 1

        return state

    def _dampen_down(self, min_hold_time=0.5):
        action = np.zeros(
            shape=self.action_space.shape,
            dtype=self.action_space.dtype)

        time_hold = min_hold_time * self._frequency
        samples_downwards = 0  # Consecutive samples pendulum is stationary

        while True:
            state, _, _, _ = self.step(action)
            # Break if pendulum is stationary
            ref_state = [0., 0., 0., 0.]
            if np.allclose(state[4:8], ref_state, rtol=1e-02, atol=1e-03):
                if samples_downwards > time_hold:
                    break
                samples_downwards += 1
            else:
                samples_downwards = 0

        return self._get_state()

    def flip_up(self):
        return self._flip_up()

    def dampen_down(self):
        return self._dampen_down()

    def reset(self):
        # Start the pendulum stationary at the bottom (stable point)
        return self.dampen_down()

    def step(self, action):
        state = self._step(action)
        reward = self.reward_fn(state, action)
        done = False
        info = {}
        return state, reward, done, info

    def render(self, mode):
        pass

    def close(self, type=None, value=None, traceback=None):
        # Safely close the Qube
        self.qube.__exit__(type=type, value=value, traceback=traceback)
