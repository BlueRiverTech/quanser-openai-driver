from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import time
import math
import numpy as np

from gym import spaces
from gym.utils import seeding
from gym_brt.quanser import QubeSimulator

# from gym_brt.quanser import QubeHardware, QubeSimulator
from gym_brt.envs.rendering import QubeRenderer


MAX_MOTOR_VOLTAGE = 3
ACT_MAX = np.asarray([MAX_MOTOR_VOLTAGE], dtype=np.float64)
# OBS_MAX = [theta, alpha, theta_dot, alpha_dot]
OBS_MAX = np.asarray([np.pi / 2, np.pi, np.inf, np.inf], dtype=np.float64)


class QubeBaseReward(object):
    def __init__(self):
        self.target_space = spaces.Box(low=-ACT_MAX, high=ACT_MAX, dtype=np.float32)

    def __call__(self, state, action):
        raise NotImplementedError


class QubeBaseEnv(gym.Env):
    """A base class for all qube-based environments."""

    def __init__(
        self,
        frequency=1000,
        batch_size=2048,
        use_simulator=False,
        encoder_reset_steps=100000,
    ):
        self.observation_space = spaces.Box(-OBS_MAX, OBS_MAX)
        self.action_space = spaces.Box(-ACT_MAX, ACT_MAX)
        self.reward_fn = QubeBaseReward()

        self._frequency = frequency
        # Ensures that samples in episode are the same as batch size
        # Reset every batch_size steps (2048 ~= 8.192 seconds)
        self._max_episode_steps = batch_size
        self._episode_steps = 0
        self._encoder_reset_steps = encoder_reset_steps
        self._steps_since_encoder_reset = 0
        self._isdone = True
        self._target_angle = 0

        self._theta, self._alpha, self._theta_dot, self._alpha_dot = 0, 0, 0, 0
        self._dtheta, self._dalpha = 0, 0

        # Open the Qube
        if use_simulator:
            # TODO: Check assumption: ODE integration should be ~ once per ms
            integration_steps = int(np.ceil(1000 / self._frequency))
            self.qube = QubeSimulator(
                forward_model="ode",
                frequency=self._frequency,
                integration_steps=integration_steps,
                max_voltage=MAX_MOTOR_VOLTAGE,
            )
        else:
            self.qube = QubeHardware(
                frequency=self._frequency, max_voltage=MAX_MOTOR_VOLTAGE
            )
        self.qube.__enter__()

        self.seed()
        self._viewer = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close(type=type, value=value, traceback=traceback)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action, led=None):
        if led is None:
            if self._isdone:  # Doing reset
                led = [1.0, 1.0, 0.0]  # Yellow
            else:
                if abs(self._alpha) > (20 * np.pi / 180):
                    led = [1.0, 0.0, 0.0]  # Red
                elif abs(self._theta) > (90 * np.pi / 180):
                    led = [1.0, 0.0, 0.0]  # Red
                else:
                    led = [0.0, 1.0, 0.0]  # Green

        action = np.clip(np.array(action, dtype=np.float64), -ACT_MAX, ACT_MAX)
        state = self.qube.step(action, led=led)
        self._dtheta, self._dalpha = state[0] - self._theta, state[1] - self._alpha
        self._theta, self._alpha, self._theta_dot, self._alpha_dot = state

    def _get_state(self):
        state = np.array(
            [self._theta, self._alpha, self._theta_dot, self._alpha_dot],
            dtype=np.float64,
        )
        return state

    def reset(self):
        self._episode_steps = 0
        # Occasionaly reset the enocoders to remove sensor drift
        if self._steps_since_encoder_reset >= self._encoder_reset_steps:
            self.qube.reset_encoders()
            self._steps_since_encoder_reset = 0

        action = np.zeros(shape=self.action_space.shape, dtype=self.action_space.dtype)
        self._step(action)
        return self._get_state()

    def _reset_up(self):
        self.qube.reset_up()
        action = np.zeros(shape=self.action_space.shape, dtype=self.action_space.dtype)
        self._step(action)
        return self._get_state()

    def _reset_down(self):
        self.qube.reset_down()
        action = np.zeros(shape=self.action_space.shape, dtype=self.action_space.dtype)
        self._step(action)
        return self._get_state()

    def step(self, action):
        self._step(action)
        state = self._get_state()

        theta, alpha, theta_dot, alpha_dot = state
        reward = 1 - (
            (0.8 * np.abs(alpha) + 0.2 * np.abs(self._target_angle - theta)) / np.pi
        )

        self._episode_steps += 1
        self._steps_since_encoder_reset += 1

        done = False
        done |= self._episode_steps % self._max_episode_steps == 0
        done |= abs(theta) > (90 * np.pi / 180)
        self._isdone = done

        info = {
            "theta": theta,
            "alpha": alpha,
            "theta_dot": theta_dot,
            "alpha_dot": alpha_dot,
        }
        return state, reward, self._isdone, info

    def render(self, mode="human"):
        if self._viewer is None:
            self._viewer = QubeRenderer(self._theta, self._alpha, self._frequency)
        self._viewer.render(self._theta, self._alpha)

    def close(self, type=None, value=None, traceback=None):
        # Safely close the Qube (important on hardware)
        self.qube.close(type=type, value=value, traceback=traceback)
        if self._viewer is not None:
            self._viewer.close()
