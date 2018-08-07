from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import numpy as np
from gym import spaces
from gym_brt.envs.qube_base_env import \
    QubeBaseEnv, \
    normalize_angle, \
    MAX_MOTOR_VOLTAGE, \
    ACTION_HIGH, \
    ACTION_LOW, \
    WARMUP_TIME
from gym_brt.control import QubeFlipUpInvertedClassicControl


# Time given to the classical control system to flip up before quitting
FLIP_UP_TIME_OUT = np.inf
# Time in seconds to hold the pendulum upright within a tolerance
MIN_INVERTED_HOLD_TIME = 1
# Angle from perfectly inverted that counts as 'upright' (in radians)
ALPHA_TOLERANCE = 10 * np.pi / 180


class QubeHoldInvertedReward(object):
    def __init__(self):
        self.target_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH, dtype=np.float32)

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
        alpha = np.arctan2(alpha_y, alpha_x)  # pole

        cost = normalize_angle(theta)**4 + \
            normalize_angle(alpha)**2 + \
            0.1 * alpha_velocity**2

        reward = -cost
        return reward


class QubeHoldInvertedEnv(QubeBaseEnv):
    def __init__(self, env_base='QubeServo2', frequency=1000, alpha_tolerance=None):
        super(QubeHoldInvertedEnv, self).__init__(env_base=env_base, frequency=frequency)
        self.reward_fn = QubeHoldInvertedReward()
        self._alpha_tolerance = alpha_tolerance if alpha_tolerance is not None \
                                                else ALPHA_TOLERANCE

    def reset(self):
        # Start the pendulum stationary at the bottom (stable point)
        if WARMUP_TIME > 0:
            start_time = time.time()
            while (time.time() - start_time) < WARMUP_TIME:
                action = np.zeros(
                    shape=self.action_space.shape,
                    dtype=self.action_space.dtype)
                state = self._step(action)
        else:
            action = np.zeros(
                shape=self.action_space.shape,
                dtype=self.action_space.dtype)
            state = self._step(action)

        # Run classic control for flip-up until the pendulum is inverted for a
        # set amount of time
        control = QubeFlipUpInvertedClassicControl(self,
            sample_freq=self._frequency)
        time_out_samples = FLIP_UP_TIME_OUT * self._frequency
        time_hold_samples = MIN_INVERTED_HOLD_TIME * self._frequency

        total_samples = 0 # Number of samples since control system started
        samples_upright = 0 # Consecutive samples that the pendulum has been inverted 
        while True:
            action = control.action(state)
            state, _, _, _ = self.step(action)

            # Break if timed out
            if total_samples > time_out_samples:
                print('Warning: flip-up control in reset timed out.')
                break

            # Break if pendulum is inverted
            if self._alpha < self._alpha_tolerance:
                if samples_upright > time_hold_samples:
                    break
                samples_upright += 1
            else:
                samples_upright = 0
            total_samples += 1

        return state

    def _done(self):
        # The episode ends whenever the angle alpha is outside the tolerance
        return self._alpha > self._alpha_tolerance

    def step(self, action):
        state, reward, _, info = super(QubeHoldInvertedEnv, self).step(action)
        done = self._done()
        return state, reward, done, info



def main():
    num_episodes = 10
    num_steps = 250

    with QubeHoldInvertedEnv() as env:
        for episode in range(num_episodes):
            state = env.reset()
            for step in range(num_steps):
                action = env.action_space.sample()
                state, reward, done, _ = env.step(action)


if __name__ == '__main__':
    main()