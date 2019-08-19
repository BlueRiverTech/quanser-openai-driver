from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from gym import spaces
from gym_brt.envs.qube_base_env import QubeBaseEnv


class QubeBalanceEnv(QubeBaseEnv):
    """
    Description:
        A pendulum is attached to an un-actuated joint to a horizontal arm,
        which is actuated by a rotary motor. The pendulum begins
        upright/inverted and the goal is to keep it from falling by applying a
        voltage on the motor which causes a torque on the horizontal arm.

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
        1   Pendulum angle (alpha)       -20 deg      20 deg
        2   Cart Velocity                -Inf         Inf
        3   Pole Velocity                -Inf         Inf
        Note: the velocities are limited by the physical system.

    Actions:
        Type: Real number (1-D Continuous) (voltage applied to motor)

    Reward:
        r(s_t, a_t) = 1 + 0.8 * np.cos(alpha) + 0.2 * np.cos(theta)

    Starting State:
        Use a classical controller to get the pendulum into it's initial
        inverted state. This has inherent randomness.

    Episode Termination:
        Pendulum Angle (alpha) is greater than ±20° from upright, when theta is
        greater than ±90°, or after 2048 steps
    """

    def reset(self):
        super(QubeBalanceEnv, self).reset()
        # Start the pendulum stationary at the top (stable point)
        state = self._reset_up()
        return state

    def step(self, action):
        state, reward, done, info = super(QubeBalanceEnv, self).step(action)
        theta, alpha = state[:2]
        self._isdone |= abs(alpha) > (20 * np.pi / 180)
        return state, reward, self._isdone, info


def target_angle():
    max_angle = 80 * (np.pi / 180)  # 80 degrees
    return np.random.uniform(-max_angle, max_angle)


class QubeBalanceFollowEnv(QubeBalanceEnv):
    def __init__(self, frequency=250, **kwargs):
        super(QubeBalanceFollowEnv, self).__init__(frequency=frequency, **kwargs)
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
        state = super(QubeBalanceFollowEnv, self).reset()
        self._target_angle = target_angle()
        return state
