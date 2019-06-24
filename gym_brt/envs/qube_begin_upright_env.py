from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from gym import spaces
from gym_brt.envs.qube_base_env import QubeBaseEnv


class QubeBeginUprightReward(object):
    def __init__(self):
        pass

    def __call__(self, state, action):
        theta_x, theta_y = state[0], state[1]
        alpha_x, alpha_y = state[2], state[3]
        theta = np.arctan2(theta_y, theta_x)
        alpha = np.arctan2(alpha_y, alpha_x)
        # Encourage alpha=0, theta=0
        return 1 - 0.5 * (np.abs(alpha) + np.abs(theta))


class QubeBeginUprightEnv(QubeBaseEnv):
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
        0   Cos Arm angle (theta)        -1.0         1.0
        1   Sin Arm angle (theta)        -1.0         1.0
        2   Cos Pendulum angle (theta)   -1.0         1.0
        3   Sin Pendulum angle (theta)   -1.0         1.0
        4   Cart Velocity                -Inf         Inf
        5   Pole Velocity                -Inf         Inf
        Note: the velocities are limited by the physical system.

    Actions:
        Type: Real number (1-D Continuous) (voltage applied to motor)

    Reward:
        Reward is 1 - abs(alpha) - 0.1*abs(theta) for every step taken,
        including the termination step. This encourages the pendulum to stay
        stationary and stable at the top.

    Starting State:
        Use a classical controller to get the pendulum into it's initial
        inverted state. This has inherent randomness.

    Episode Termination:
        Pendulum Angle (alpha) is greater than ±20° from upright, when theta is
        greater than ±90°, or after 2048 steps
    """

    def __init__(self, frequency=250, **kwargs):
        super(QubeBeginUprightEnv, self).__init__(frequency=frequency, **kwargs)
        self.reward_fn = QubeBeginUprightReward()

    def reset(self):
        super(QubeBeginUprightEnv, self).reset()
        # Start the pendulum stationary at the top (stable point)
        state = self._reset_up()
        return state

    def step(self, action):
        state, reward, done, info = super(QubeBeginUprightEnv, self).step(action)
        self._isdone |= abs(info["alpha"]) > (20 * np.pi / 180)

        return state, reward, self._isdone, info
