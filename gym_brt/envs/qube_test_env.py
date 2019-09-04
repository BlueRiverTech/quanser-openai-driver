from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from gym import spaces
from gym_brt.envs.qube_base_env import QubeBaseEnv
from gym_brt.envs.qube_swingup_env import QubeSwingupFollowEnv
from gym_brt.envs.qube_balance_env import QubeBalanceFollowEnv
from gym_brt.envs.qube_dampen_env import QubeDampenFollowEnv
from gym_brt.envs.qube_rotor_env import QubeRotorFollowEnv


"""
A series of testing evironments to visualize what a policy learned in each of
the `Follow` environments.
"""


class QubeBalanceFollowSineWaveEnv(QubeBalanceFollowEnv):
    def _next_target_angle(self):
        # Sine wave between -180 to +180 degrees
        print(
            "T:{:05.2f}, C:{:05.2f}, Diff:{:05.2f}".format(
                self._target_angle * (57.3),
                57.3 * self._theta,
                57.3 * (self._target_angle - self._theta),
            )
        )
        return np.pi / 3 * np.sin(self._episode_steps / self._frequency)


class QubeSwingupFollowSineWaveEnv(QubeSwingupFollowEnv):
    def _next_target_angle(self):
        # Sine wave between -180 to +180 degrees
        print(
            "T:{:05.2f}, C:{:05.2f}, Diff:{:05.2f}".format(
                self._target_angle * (57.3),
                57.3 * self._theta,
                57.3 * (self._target_angle - self._theta),
            )
        )
        return np.pi / 3 * np.sin(self._episode_steps / self._frequency)


class QubeRotorFollowSineWaveEnv(QubeRotorFollowEnv):
    def _next_target_angle(self):
        # Sine wave between -180 to +180 degrees
        print(
            "T:{:05.2f}, C:{:05.2f}, Diff:{:05.2f}".format(
                self._target_angle * (57.3),
                57.3 * self._theta,
                57.3 * (self._target_angle - self._theta),
            )
        )
        return np.pi / 3 * np.sin(self._episode_steps / self._frequency)


class QubeDampenFollowSineWaveEnv(QubeDampenFollowEnv):
    def _next_target_angle(self):
        # Sine wave between -180 to +180 degrees
        print(
            "T:{:05.2f}, C:{:05.2f}, Diff:{:05.2f}".format(
                self._target_angle * (57.3),
                57.3 * self._theta,
                57.3 * (self._target_angle - self._theta),
            )
        )
        return np.pi * np.sin(self._episode_steps / self._frequency)
