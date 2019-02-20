from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from gym import spaces
from gym_brt.envs.qube_base_env import \
    QubeBaseEnv, \
    normalize_angle, \
    ACTION_HIGH, \
    ACTION_LOW


OBSERVATION_HIGH = np.asarray([
    1, 1, 1, 1,  # cos/sin of angles
    np.inf, np.inf,  # velocities
], dtype=np.float64)
OBSERVATION_LOW = -OBSERVATION_HIGH


class QubeBeginUprightReward(object):
    def __init__(self):
        self.target_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH, dtype=np.float32)

        self.buffer = 0
        self.prev_energy = 0

    def __call__(self, state, action):
        theta_x, theta_y = state[0], state[1]
        alpha_x, alpha_y = state[2], state[3]
        theta = np.arctan2(theta_y, theta_x)
        alpha = np.arctan2(alpha_y, alpha_x)
        theta_dot = state[4]
        alpha_dot = state[5]

        return 1 - np.abs(alpha) - 0.1*np.abs(theta)


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
        greater than ±90°, or after 5 seconds have past
    """
    def __init__(self, frequency=300, use_simulator=False):
        super(QubeBeginUprightEnv, self).__init__(
            frequency=frequency,
            use_simulator=use_simulator)
        self.reward_fn = QubeBeginUprightReward()

        self.observation_space = spaces.Box(
            OBSERVATION_LOW, OBSERVATION_HIGH,
            dtype=np.float32)

        self._episode_steps = 0

    def reset(self):
        # Start the pendulum stationary at the top (stable point)
        state = self.flip_up()
        self._episode_steps = 0
        return state

    def _done(self):
        # The episode ends whenever the angle alpha is outside the tolerance
        if abs(self._alpha) > (20 * np.pi / 180):
            return True
        elif abs(self._theta) > (90 * np.pi / 180):
            return True
        elif self._episode_steps > (self._frequency * 5):  # Greater than 5s
            return True
        else:
            return False

    def step(self, action):
        state, reward, _, info = super(QubeBeginUprightEnv, self).step(action)
        # Make the state simpler/closer to cartpole
        state = state[:6]
        done = self._done()
        self._episode_steps += 1
        return state, reward, done, info


def main():
    num_episodes = 10
    num_steps = 250

    with QubeBeginUprightEnv() as env:
        for episode in range(num_episodes):
            state = env.reset()
            for step in range(num_steps):
                action = env.action_space.sample()
                state, reward, done, _ = env.step(action)


if __name__ == '__main__':
    main()
