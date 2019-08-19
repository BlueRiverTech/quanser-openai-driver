from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import numpy as np
from gym import spaces
from gym_brt.envs.qube_base_env import QubeBaseEnv


class QubeDampenEnv(QubeBaseEnv):
    def __init__(self, frequency=250, **kwargs):
        super(QubeDampenEnv, self).__init__(frequency=frequency, **kwargs)

    def reset(self):
        super(QubeDampenEnv, self).reset()
        state = self._reset_up()
        time.sleep(np.random.randint(1, 3000) / 1000)  # Sleep between 1-3000ms
        return state

    def step(self, action):
        state, reward, done, info = super(QubeDampenEnv, self).step(action)
        theta, alpha = state[:2]
        reward = 0.5 + 0.5 * (np.abs(alpha) - np.abs(theta)) / np.pi

        return state, reward, self._isdone, info


def target_angle():
    max_angle = 80 * (np.pi / 180)  # 80 degrees
    return np.random.uniform(-max_angle, max_angle)


class QubeDampenFollowEnv(QubeDampenEnv):
    def __init__(self, frequency=250, **kwargs):
        super(QubeDampenFollowEnv, self).__init__(frequency=frequency, **kwargs)
        self._target_angle = target_angle()

    def _get_state(self):
        state = np.array(
            [
                self._theta,
                self._alpha,
                self._theta_dot,
                self._alpha_dot,
                self._target_angle,
            ],
            dtype=np.float64,
        )
        return state

    def reset(self):
        super(QubeDampenFollowEnv, self).reset()
        self._target_angle = target_angle()
        return state

    def step(self, action):
        state, reward, done, info = super(QubeDampenFollowEnv, self).step(action)
        theta, alpha = state[:2]
        reward = (
            0.5 + 0.5 * (np.abs(alpha) - np.abs(self._target_angle - theta)) / np.pi
        )

        return state, reward, self._isdone, info
