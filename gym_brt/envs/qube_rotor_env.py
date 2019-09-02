from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from gym import spaces
from gym_brt.envs.qube_base_env import QubeBaseEnv


class QubeRotorEnv(QubeBaseEnv):
    def _reward(self):
        # Few options for reward:
        # - high reward for large alpha_dot and small theta
        # - reward for matching the action of the RPM controller
        # - reward for how close the pendulum matches a clock hand going at a certain RPM

        theta_dist = (1 - np.abs(self._target_angle - self._theta)) / np.pi
        return np.clip(theta_dist * self._alpha_dot / 20, 0, 20)

    def _isdone(self):
        done = False
        done |= self._episode_steps >= self._max_episode_steps
        done |= abs(self._theta) > (90 * np.pi / 180)
        return done

    def reset(self):
        super(QubeRotorEnv, self).reset()
        state = self._reset_down()
        return state


class QubeRotorFollowEnv(QubeRotorEnv):
    def __init__(self, **kwargs):
        super(QubeRotorFollowEnv, self).__init__(**kwargs)
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
