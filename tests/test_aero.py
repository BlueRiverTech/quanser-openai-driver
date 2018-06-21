from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from gym_brt import AeroPositionEnv
import gym
import time


STATE_KEYS = [ 
        'PITCH',
        'YAW',
        'VELOCITY_X',
        'VELOCITY_Y',
        'VELOCITY_Z',
        'ACCELERATION_X',
        'ACCELERATION_Y',
        'ACCELERATION_Z',
        'TACH_PITCH',
        'TACH_YAW',
        'SENSE0',
        'SENSE1'
        ]

def print_info(state, action, reward):
    print("State:")
    for name, val in zip(STATE_KEYS, state):
        print("\t{}: {}".format(name, val))
    print("\nAction: {}".format(action))
    print("Reward: {}".format(reward))


def test_env(env_name, action_func=None):
    num_episodes = 10
    num_steps = 250

    with gym.make(env_name) as env:
        for episode in range(num_episodes):
            state = env.reset()
            for step in range(num_steps):
                if action_func is None:
                    action = env.action_space.sample()
                else:
                    action = action_func(state)
                next_state, reward, done, _ = env.step(action)
                if done:
                    break
                state = next_state

                rand_time = 0.1 * np.random.rand()
                time.sleep(1)
                print_info(state, action, reward)

    """
    # Another way to run the Qube enviroment
    env = gym.make('Qube-v0')

    num_episodes = 10
    num_steps = 250
    try:
        for episode in range(num_episodes):
            state = env.reset()
            for step in range(num_steps):
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                if done:
                    break
                state = next_state
    finally:
        # Note: to set all encoders and motor voltages to 0, you must call env.close()
        env.close()
    """


if __name__ == '__main__':
    print('Testing Aero Position Env')
    test_env('Aero-Position-v0')
