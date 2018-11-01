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


class QubeBeginDownReward(object):
    def __init__(self):
        self.target_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH, dtype=np.float32)

    def __call__(self, state, action):
        alpha_x, alpha_y = state[2], state[3]
        alpha = np.arctan2(alpha_y, alpha_x)

        reward = (abs(alpha) < (20 * np.pi / 180))
        return reward


class QubeBeginDownEnv(QubeBaseEnv):
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
        Reward is 0 for when the pendulum is not upright (alpha is greater than
        ±20°) and the reward is 1 for every step taken where the pendulum is
        upright (alpha is smaller in magnitude than ±20°)

    Starting State:
        Use a classical controller to get the pendulum into it's initial
        inverted state. This has inherent randomness.

    Episode Termination:
        After 5 seconds.
    """
    def __init__(self, frequency=300, use_simulator=False):
        super(QubeBeginDownEnv, self).__init__(
            frequency=frequency,
            use_simulator=use_simulator)
        self.reward_fn = QubeBeginDownReward()

        self._episode_steps = 0

    def reset(self):
        # Start the pendulum stationary at the top (stable point)
        self._episode_steps = 0
        return super(QubeBeginDownEnv, self).reset()

    def _done(self):
        done = self._episode_steps > (self.frequency * 5) # Greater than 5s
        return done

    def step(self, action):
        state, reward, _, info = super(QubeBeginDownEnv, self).step(action)
        # Make the state simpler/closer to cartpole
        state = state[:6]
        done = self._done()
        self._episode_steps += 1
        return state, reward, done, info


def main():
    num_episodes = 10
    num_steps = 250

    with QubeBeginDownEnv() as env:
        for episode in range(num_episodes):
            state = env.reset()
            for step in range(num_steps):
                action = env.action_space.sample()
                state, reward, done, _ = env.step(action)


if __name__ == '__main__':
    main()
