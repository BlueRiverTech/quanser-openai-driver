from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import time
import math
import numpy as np

from gym import spaces
from gym.utils import seeding
from gym_brt.quanser import QubeHardware, QubeSimulator


MAX_MOTOR_VOLTAGE = 3
ACT_MAX = np.asarray([MAX_MOTOR_VOLTAGE], dtype=np.float64)
# OBS_MAX = [cos(theta), sin(theta), cos(alpha), sin(alpha), theta_dot, alpha_dot]
OBS_MAX = np.asarray([1, 1, 1, 1, np.inf, np.inf], dtype=np.float64)


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
        self._theta, self._alpha, self._theta_dot, self._alpha_dot = state

    def _get_state(self):
        state = np.array(
            [
                np.cos(self._theta),
                np.sin(self._theta),
                np.cos(self._alpha),
                np.sin(self._alpha),
                self._theta_dot,
                self._alpha_dot,
            ],
            dtype=np.float64,
        )
        return state

    def reset(self):
        self._episode_steps = 0
        # Occasionaly reset the enocoders to remove sensor drift
        if self._steps_since_encoder_reset >= self._encoder_reset_steps:
            self.hard_reset()
            self._steps_since_encoder_reset = 0

        action = np.zeros(shape=self.action_space.shape, dtype=self.action_space.dtype)
        return self._step(action)

    def _reset_up(self):
        return self.qube.reset_up()

    def _reset_down(self):
        return self.qube.reset_down()

    def step(self, action):
        self._step(action)
        state = self._get_state()

        reward = self.reward_fn(state, action)

        self._episode_steps += 1
        self._steps_since_encoder_reset += 1

        done = False
        done |= self._episode_steps % self._max_episode_steps == 0
        done |= abs(self._theta) > (90 * np.pi / 180)
        self._isdone = done

        info = {
            "theta": self._theta,
            "alpha": self._alpha,
            "theta_dot": self._theta_dot,
            "alpha_dot": self._alpha_dot,
        }
        return state, reward, self._isdone, info

    def render(self, mode="human"):
        if self._viewer is None:
            from gym.envs.classic_control import rendering

            width, height = (640, 240)
            self._viewer = rendering.Viewer(width, height)
            l, r, t, b = (2, -2, 0, 100)
            theta_poly = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
            l, r, t, b = (2, -2, 0, 100)
            alpha_poly = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
            theta_circle = rendering.make_circle(radius=100, res=64, filled=False)
            theta_circle.set_color(0.5, 0.5, 0.5)  # Theta is grey
            alpha_circle = rendering.make_circle(radius=100, res=64, filled=False)
            alpha_circle.set_color(0.8, 0.0, 0.0)  # Alpha is red
            theta_origin = (width / 2 - 150, height / 2)
            alpha_origin = (width / 2 + 150, height / 2)
            self._theta_tx = rendering.Transform(translation=theta_origin)
            self._alpha_tx = rendering.Transform(translation=alpha_origin)
            theta_poly.add_attr(self._theta_tx)
            alpha_poly.add_attr(self._alpha_tx)
            theta_circle.add_attr(self._theta_tx)
            alpha_circle.add_attr(self._alpha_tx)
            self._viewer.add_geom(theta_poly)
            self._viewer.add_geom(alpha_poly)
            self._viewer.add_geom(theta_circle)
            self._viewer.add_geom(alpha_circle)

        self._theta_tx.set_rotation(self._theta + np.pi)
        self._alpha_tx.set_rotation(self._alpha)

        return self._viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self, type=None, value=None, traceback=None):
        # Safely close the Qube (important on hardware)
        self.qube.close(type=type, value=value, traceback=traceback)
        if self._viewer:
            self._viewer.close()
