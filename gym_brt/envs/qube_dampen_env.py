from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import numpy as np
from gym import spaces
from gym_brt.envs.qube_base_env import QubeBaseEnv


class QubeDampenEnv(QubeBaseEnv):
    def _reward(self):
        alpha = self._alpha
        alpha = (alpha + 2 * np.pi) % (2 * np.pi) - np.pi
        reward = 1 - (
            (0.8 * np.abs(alpha) + 0.2 * np.abs(self._target_angle - self._theta))
            / np.pi
        )
        return max(reward, 0)  # Clip for the follow env case

    def _isdone(self):
        done = False
        done |= self._episode_steps >= self._max_episode_steps
        done |= abs(self._theta) > (90 * np.pi / 180)
        return done

    def reset(self):
        super(QubeDampenEnv, self).reset()
        state = self._reset_up()
        time.sleep(np.random.randint(1, 3000) / 1000)  # Sleep between 1-3000ms
        return state


class QubeDampenSparseEnv(QubeDampenEnv):
    def _reward(self):
        within_range = True
        # Assume -pi < alpha < pi
        within_range &= np.abs(self._alpha) > (189 * np.pi / 180)
        within_range &= np.abs(self._theta) < (1 * np.pi / 180)
        return 1 if within_range else 0


class QubeDampenFollowEnv(QubeDampenEnv):
    def __init__(self, **kwargs):
        super(QubeDampenFollowEnv, self).__init__(**kwargs)
        obs_max = np.asarray(
            [np.pi / 2, np.pi, np.inf, np.inf, 80 * (np.pi / 180)], dtype=np.float64
        )
        self.observation_space = spaces.Box(-obs_max, obs_max)

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

    def _next_target_angle(self):
        # Update the target angle twice a second on average at random intervals
        if np.random.randint(1, self._frequency / 2) == 1:
            max_angle = 80 * (np.pi / 180)  # 80 degrees
            angle = np.random.uniform(-max_angle, max_angle)
        else:
            angle = self._target_angle
        return angle


class QubeDampenFollowSparseEnv(QubeDampenFollowEnv):
    def _reward(self):
        within_range = True
        # Assume -pi < alpha < pi
        within_range &= np.abs(self._alpha) > (189 * np.pi / 180)
        within_range &= np.abs(self._theta) < (1 * np.pi / 180)
