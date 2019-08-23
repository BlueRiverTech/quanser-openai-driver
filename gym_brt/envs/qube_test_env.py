from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from gym import spaces
from gym_brt.envs.qube_base_env import QubeBaseEnv


"""
A series of testing evironments to visualize what a policy learned in each of
the `Follow` environments.
"""


class QubeBalanceFollowSineWaveEnv(QubeBaseEnv):
    def _get_state(self):
        return np.array(
            [
                self._theta - self._next_target_angle(),
                self._alpha,
                self._theta_dot,
                self._alpha_dot,
            ],
            dtype=np.float64,
        )

    def _reward(self):
        return 0

    def _isdone(self):
        return abs(self._theta) > np.pi / 2

    def reset(self):
        super(QubeBalanceFollowSineWaveEnv, self).reset()
        state = self._reset_up()
        return state

    def _next_target_angle(self):
        # Sine wave between -180 to +180 degrees
        return np.pi / 3 * np.cos(self._episode_steps / self._frequency)


class QubeSwingupFollowSineWaveEnv(QubeBaseEnv):
    def _get_state(self):
        target_angle = (
            self._next_target_angle() if self._alpha < (20 * np.pi / 180) else 0
        )
        return np.array(
            [self._theta - target_angle, self._alpha, self._theta_dot, self._alpha_dot],
            dtype=np.float64,
        )

    def _reward(self):
        return 0

    def _isdone(self):
        return abs(self._theta) > np.pi / 2

    def reset(self):
        super(QubeSwingupFollowSineWaveEnv, self).reset()
        state = self._reset_down()
        return state

    def _next_target_angle(self):
        # Sine wave between -180 to +180 degrees
        return np.pi / 3 * np.cos(self._episode_steps / self._frequency)


class QubeRotorFollowSineWaveEnv(QubeSwingupFollowSineWaveEnv):
    pass


class QubeDampenFollowSineWaveEnv(QubeBaseEnv):
    def _reward(self):
        return 0

    def _isdone(self):
        return False

    def reset(self):
        super(QubeBalanceFollowSineWaveEnv, self).reset()
        state = self._reset_up()
        time.sleep(np.random.randint(1, 3000) / 1000)  # Sleep between 1-3000ms
        return state

    def _next_target_angle(self):
        # Sine wave between -180 to +180 degrees
        return np.pi * np.cos(self._episode_steps / self._frequency)
