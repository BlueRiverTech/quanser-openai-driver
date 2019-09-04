from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from gym import spaces
from gym_brt.envs.qube_base_env import QubeBaseEnv

"""
    Description:
        A pendulum is attached to an un-actuated joint to a horizontal arm,
        which is actuated by a rotary motor. The pendulum begins
        downwards and the goal is flip the pendulum up and then to keep it from
        falling by applying a voltage on the motor which causes a torque on the
        horizontal arm.

    Source:
        This is modified for the Quanser Qube Servo2-USB from the Cart Pole
        problem described by Barto, Sutton, and Anderson, and implemented in
        OpenAI Gym: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        This description is also modified from the description by the OpenAI
        team.

    Observation:
        Type: Box(4)
        Num Observation                   Min         Max
        0   Rotary arm angle (theta)     -90 deg      90 deg
        1   Pendulum angle (alpha)       -180 deg     180 deg
        2   Cart Velocity                -Inf         Inf
        3   Pole Velocity                -Inf         Inf
        Note: the velocities are limited by the physical system.

    Actions:
        Type: Real number (1-D Continuous) (voltage applied to motor)

    Reward:
        r(s_t, a_t) = 1 - (0.8 * abs(alpha) + 0.2 * abs(theta)) / pi

    Starting State:
        Theta = 0 + noise, alpha = pi + noise

    Episode Termination:
        When theta is greater than ±90° or after 2048 steps
"""


class QubeSwingupEnv(QubeBaseEnv):
    def _reward(self):
        reward = 1 - (
            (0.8 * np.abs(self._alpha) + 0.2 * np.abs(self._target_angle - self._theta))
            / np.pi
        )
        return max(reward, 0)  # Clip for the follow env case

    def _isdone(self):
        done = False
        done |= self._episode_steps >= self._max_episode_steps
        done |= abs(self._theta) > (90 * np.pi / 180)
        return done

    def reset(self):
        super(QubeSwingupEnv, self).reset()
        state = self._reset_down()
        return state


class QubeSwingupSparseEnv(QubeSwingupEnv):
    def _reward(self):
        within_range = True
        within_range &= np.abs(self._alpha) < (1 * np.pi / 180)
        within_range &= np.abs(self._theta) < (1 * np.pi / 180)
        return 1 if within_range else 0


class QubeSwingupFollowEnv(QubeSwingupEnv):
    def __init__(self, **kwargs):
        super(QubeSwingupFollowEnv, self).__init__(**kwargs)
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


class QubeSwingupFollowSparseEnv(QubeSwingupFollowEnv):
    def _reward(self):
        within_range = True
        within_range &= np.abs(self._alpha) < (1 * np.pi / 180)
        within_range &= np.abs(self._theta) < (1 * np.pi / 180)
        return 1 if within_range else 0
