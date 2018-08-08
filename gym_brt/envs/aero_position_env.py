from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import numpy as np

from gym import spaces
from gym.utils import seeding
from gym_brt.envs.QuanserWrapper import QuanserAero


# pitch, yaw, gyro, acceleration, current sense
OBSERVATION_HIGH = np.asarray([
    2048,  # pitch
    np.inf,  # yaw (technically unlimited...)
    np.pi / 4, np.pi / 4, np.pi / 4,  # gyro/velocity (x,y,z)
    np.pi / 4, np.pi / 4, np.pi / 4,  # acceleration (x,y,z)
    4100,  # tach for pitch
    4100,  # tach for yaw
    2., 2.,  # peak current
], dtype=np.float64)

OBSERVATION_LOW = -OBSERVATION_HIGH

MAX_MOTOR_VOLTAGE = 20.0
ACTION_HIGH = np.asarray(
    [MAX_MOTOR_VOLTAGE, MAX_MOTOR_VOLTAGE],
    dtype=np.float64)
ACTION_LOW = -ACTION_HIGH

WARMUP_STEPS = 100


STATE_KEYS = [
    'PITCH',
    'YAW',
    'VELOCITY_X',
    'VELOCITY_Y',
    'VELOCITY_Z',
    'ACCELERATION_X',
    'ACCELERATION_Y',
    'ACCELERATION_Z',
    'TACH_PITCH',
    'TACH_YAW',
    'SENSE0',
    'SENSE1'
]


def normalize_angle(theta):
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


class AeroPositionReward(object):
    def __init__(self, *args):
        self.target_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH, dtype=np.float32)

    def __call__(self, state, action):
        pitch = state[0]
        yaw = state[1]
        velocity_x = state[2]
        velocity_y = state[3]
        velocity_z = state[4]
        acceleration_x = state[4]
        acceleration_y = state[5]
        acceleration_z = state[6]

        cost = pitch**2 + \
            yaw**2 + \
            0.01 * velocity_x**2 + \
            0.01 * velocity_y**2 + \
            0.01 * velocity_z**2 + \
            0.01 * acceleration_x**2 + \
            0.01 * acceleration_y**2 + \
            0.01 * acceleration_z**2

        reward = -cost
        return reward


class AeroPositionEnv(gym.Env):
    def __init__(self, frequency=1000):
        self.observation_space = spaces.Box(
            OBSERVATION_LOW, OBSERVATION_HIGH,
            dtype=np.float32)

        self.action_space = spaces.Box(
            ACTION_LOW, ACTION_HIGH,
            dtype=np.float32)

        self.reward_fn = AeroPositionReward()

        self._pitch = 0
        self._yaw = 0
        self._velocity_x = 0
        self._velocity_y = 0
        self._velocity_z = 0
        self._acceleration_x = 0
        self._acceleration_y = 0
        self._acceleration_z = 0
        self._sense0 = 0
        self._sense1 = 0

        # Open the Aero
        self.aero = QuanserAero(frequency=frequency)
        self.aero.__enter__()

        self.seed()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        motor_voltages = np.clip(
            np.array([action[0], action[1]], dtype=np.float64),
            ACTION_LOW, ACTION_HIGH)
        currents, encoders, others = self.aero.action(motor_voltages)

        self._pitch = encoders[2] * ((2 * np.pi) / 2048.0)  # In rads
        self._yaw = encoders[3] * ((2 * np.pi) / 4096.0)  # In rads

        self._velocity_x = others[0]
        self._velocity_y = others[1]
        self._velocity_z = others[2]
        self._acceleration_x = others[3]
        self._acceleration_y = others[4]
        self._acceleration_z = others[5]
        self._sense0 = currents[0]
        self._sense1 = currents[1]

        return self._get_state()

    def _get_state(self):
        state = np.asarray([
            self._pitch,
            self._yaw,
            self._velocity_x,
            self._velocity_y,
            self._velocity_z,
            self._acceleration_x,
            self._acceleration_y,
            self._acceleration_z,
            self._sense0,
            self._sense1,
        ], dtype=np.float64)
        return state

    def reset(self):
        # Step once with no action to init state
        if WARMUP_STEPS > 0:
            for i in range(WARMUP_STEPS):
                action = np.zeros(
                    shape=self.action_space.shape,
                    dtype=self.action_space.dtype)
                state = self._step(action)
            return state
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
        # Safely close the aero
        self.aero.__exit__(None, None, None)


def main():
    num_episodes = 10
    num_steps = 250

    with AeroPositionEnv() as env:
        for episode in range(num_episodes):
            state = env.reset()
            for step in range(num_steps):
                action = env.action_space.sample()
                state, reward, done, _ = env.step(action)


if __name__ == '__main__':
    main()
