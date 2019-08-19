from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from gym import spaces
from gym_brt.envs.qube_base_env import QubeBaseEnv


class QubeRotorEnv(QubeBaseEnv):
    def __init__(self, frequency=250, **kwargs):
        super(QubeRotorEnv, self).__init__(frequency=frequency, **kwargs)

    def reset(self):
        super(QubeRotorEnv, self).reset()
        state = self._reset_down()
        return state

    def step(self, action):
        state, reward, done, info = super(QubeDampenFollowEnv, self).step(action)
        theta, alpha = state[:2]

        theta, alpha, theta_dot, alpha_dot = state
        # Few options for reward:
        # - high reward for large alpha_dot and small theta
        # - reward for matching the action of the RPM controller
        # - reward for how close the pendulum matches a clock hand going at a certain RPM
        reward = 0

        return state, reward, self._isdone, info
