from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym_qube
import gym


STATE_KEYS = [ 
        'COS_THETA',
        'SIN_THETA',
        'COS_ALPHA',
        'SIN_ALPHA',
        'THETA_VELOCITY',
        'ALPHA_VELOCITY',
        'THETA_ACCELERATION',
        'ALPHA_ACCELERATION',
        'TACH0',
        'SENSE'
        ]


def print_info(state, action):
    print("\nAction: {}".format(action))
    print("State:")
    for name, val in zip(STATE_KEYS, state):
        print("\t{}: {}".format(name, val))



def main():
    num_episodes = 10
    num_steps = 250

    with gym.make('Qube-v0') as env:
        for episode in range(num_episodes):
            state = env.reset()
            for step in range(num_steps):
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                if done:
                    break
                state = next_state

                print_info(state, action)

    """
    # Another way to run the Qube enviroment
    env = QubeEnv()

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
    main()
