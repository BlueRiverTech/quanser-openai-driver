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
from gym_brt.control import QubeFlipUpControl, QubeDampenControl


MAX_MOTOR_VOLTAGE = 3
ACT_HIGH = np.asarray([MAX_MOTOR_VOLTAGE], dtype=np.float64)
ACT_LOW = -ACT_HIGH

# theta, alpha: positions, velocities, accelerations
OBS_HIGH = np.asarray([
    1, 1, 1, 1,  # cos/sin of angles
    np.inf, np.inf,  # velocities
    np.inf, np.inf,  # accelerations
    np.inf,  # tach0
    MAX_MOTOR_VOLTAGE / 8.4,  # current sense = max_voltage / R_equivalent
], dtype=np.float64)
OBS_LOW = -OBS_HIGH


class QubeBaseReward(object):
    def __init__(self):
        self.target_space = spaces.Box(
            low=ACT_LOW,
            high=ACT_HIGH, dtype=np.float32)

    def __call__(self, state, action):
        raise NotImplementedError


class QubeBaseEnv(gym.Env):
    '''A base class for all qube-based environments.'''
    def __init__(self,
                 frequency=1000,
                 batch_size=2048):
        self.observation_space = spaces.Box(OBS_LOW, OBS_HIGH)
        self.action_space = spaces.Box(ACT_LOW, ACT_HIGH)
        self.reward_fn = QubeBaseReward()

        self._theta_velocity_cstate = 0
        self._alpha_velocity_cstate = 0
        self._theta_velocity = 0
        self._alpha_velocity = 0
        self._frequency = frequency

        # Ensures that samples in episode are the same as batch size
        self._max_episode_steps = batch_size  # Reset every batch_size steps (2048 ~= 8.192 seconds)
        self._episode_steps = 0

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
        motor_voltages = np.clip(
            np.array(action, dtype=np.float64), ACT_LOW, ACT_HIGH)
        currents, encoders, others = self.qube.action(motor_voltages)

        self._sense = currents[0]
        self._tach0 = others[0]

        # Calculate alpha, theta, alpha_velocity, and theta_velocity
        self._theta = encoders[0] * (-2.0 * np.pi / 2048)
        # Alpha without normalizing
        alpha_un = encoders[1] * (2.0 * np.pi / 2048)
        # Normalized and shifted alpha
        self._alpha = (alpha_un % (2.0 * np.pi)) - np.pi

        self._theta_velocity = -2500 * self._theta_velocity_cstate + 50 * self._theta
        self._alpha_velocity = -2500 * self._alpha_velocity_cstate + 50 * alpha_un
        self._theta_velocity_cstate += (-50 * self._theta_velocity_cstate + self._theta) / self._frequency
        self._alpha_velocity_cstate += (-50 * self._alpha_velocity_cstate + alpha_un) / self._frequency

        return self._get_state()

    def _get_state(self):
        state = np.asarray([
            np.cos(self._theta),
            np.sin(self._theta),
            np.cos(self._alpha),
            np.sin(self._alpha),
            self._theta_velocity,
            self._alpha_velocity,
            self._tach0,
            self._sense,
        ], dtype=np.float32)
        return state

    def _flip_up(self):
        '''Run classic control for flip-up until the pendulum is inverted for
        a set amount of time. Assumes that initial state is stationary
        downwards.
        '''
        control = QubeFlipUpControl(env=self, sample_freq=self._frequency)
        time_hold = 1.0 * self._frequency # Number of samples to hold upright
        sample = 0 # Samples since control system started
        samples_upright = 0 # Consecutive samples pendulum is upright

        action = self.action_space.sample()
        state = self._step([1.0])
        while True:
            action = control.action(state)
            state = self._step(action)
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

        control = QubeDampenControl(env=self, sample_freq=self._frequency)
        time_hold = min_hold_time * self._frequency
        samples_downwards = 0  # Consecutive samples pendulum is stationary

        state = self._step([1.0])
        while True:
            action = control.action(state)
            state = self._step(action)

            # Break if pendulum is stationary
            ref_state = [0.]
            if abs(self._alpha) > (178 * np.pi / 180):
                if samples_downwards > time_hold:
                    break
                samples_downwards += 1
            else:
                samples_downwards = 0
        return state

    def flip_up(self, early_quit=False, time_out=5, min_hold_time=1):
        # Uncomment the following line for a more stable flip-up
        # Note: uncommenting significantly increases the flip up time
        # self.dampen_down()
        return self._flip_up()

    def dampen_down(self):
        return self._dampen_down()

    def reset(self):
        action = np.zeros(
            shape=self.action_space.shape,
            dtype=self.action_space.dtype)
        return self._step(action)

    def _done(self):
        done = False
        done |= self._episode_steps % self._max_episode_steps == 0
        done |= abs(self._theta) > (90 * np.pi / 180)
        return done

    def step(self, action):
        state = self._step(action)
        reward = self.reward_fn(state, action)

        self._episode_steps += 1

        done = self._done()
        info = {}
        return state, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self, type=None, value=None, traceback=None):
        # Safely close the Qube
        self.qube.__exit__(type=type, value=value, traceback=traceback)
