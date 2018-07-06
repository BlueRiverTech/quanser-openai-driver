from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from gym import spaces
from gym_brt.envs import QubeInvertedPendulumEnv


MAX_MOTOR_VOLTAGE = 8.0
ACTION_HIGH = np.asarray([MAX_MOTOR_VOLTAGE], dtype=np.float64)
ACTION_LOW = -ACTION_HIGH


def normalize_angle(theta):
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


class QubeInvertedPendulumSparseReward(object):

    def __init__(self):
        self.target_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, dtype=np.float32)

    def __call__(self, state, action):
        theta_x = state[0]
        theta_y = state[1]
        alpha_x = state[2]
        alpha_y = state[3]
        theta_velocity = state[4]
        alpha_velocity = state[5]
        theta_acceleration = state[6]
        alpha_acceleration = state[7]

        theta = np.arctan2(theta_y, theta_x)  # arm
        alpha = np.arctan2(alpha_y, alpha_x)  # arm

        cost = 0

        # Penalize not being upright
        if (180 / np.pi) * np.abs(normalize_angle(alpha)) > 10:
            cost += 10

        # Penalize going out of bounds
        if (180 / np.pi) * np.abs(normalize_angle(theta)) > 90:
            cost += 10

        reward = -cost
        return reward


class QubeInvertedPendulumSparseRewardEnv(QubeInvertedPendulumEnv):
    def __init__(self, frequency=1000):
        super(QubeInvertedPendulumSparseRewardEnv, self).__init__(
            frequency=frequency)
        self.reward_fn = QubeInvertedPendulumSparseReward()


def main():
    num_episodes = 10
    num_steps = 250

    with QubeInvertedPendulumSparseRewardEnv() as env:
        for episode in range(num_episodes):
            state = env.reset()
            for step in range(num_steps):
                action = env.action_space.sample()
                state, reward, done, _ = env.step(action)


if __name__ == '__main__':
    main()