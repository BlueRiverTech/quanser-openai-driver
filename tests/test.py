from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import time
import argparse
import numpy as np

from gym_brt import \
    AeroPositionEnv, \
    QubeInvertedPendulumEnv, \
    QubeInvertedPendulumSparseRewardEnv
from control import \
        NoControl, \
        RandomControl, \
        AeroClassicControl, \
        QubeHoldInvetedClassicControl

STATE_KEYS_AERO = [ 
    'PITCH         ',
    'YAW           ',
    'VELOCITY_X    ',
    'VELOCITY_Y    ',
    'VELOCITY_Z    ',
    'ACCELERATION_X',
    'ACCELERATION_Y',
    'ACCELERATION_Z',
    'TACH_PITCH    ',
    'TACH_YAW      ',
    'SENSE0        ',
    'SENSE1        '
]


STATE_KEYS_QUBE = [ 
    'COS_THETA         ',
    'SIN_THETA         ',
    'COS_ALPHA         ',
    'SIN_ALPHA         ',
    'THETA_VELOCITY    ',
    'ALPHA_VELOCITY    ',
    'THETA_ACCELERATION',
    'ALPHA_ACCELERATION',
    'TACH0             ',
    'SENSE             '
]


def print_info(state_keys, state, action, reward):
    print("State:")
    for name, val in zip(state_keys, state):
        if val < 0:
            print("\t{}: {}".format(name, val))
        else:
            print("\t{}:  {}".format(name, val))
    print("\nAction: {}".format(action))
    print("Reward: {}".format(reward))


def test_env(env_name,
             controller,
             num_episodes=10,
             num_steps=250,
             state_keys=None):

    with gym.make(env_name) as env:
        c = controller(env)
        for episode in range(num_episodes):
            state = env.reset()
            for step in range(num_steps):
                action = c.action(state)
                next_state, reward, done, _ = env.step(action)
                if done:
                    break
                state = next_state
                if state_keys is not None:
                    print_info(state_keys, state, action, reward)

    """
    # Another way to run the Qube enviroment
    env = gym.make(env_name)
    try:
        for episode in range(num_episodes):
            state = env.reset()
            for step in range(num_steps):
                action = action_func(state)
                next_state, reward, done, _ = env.step(action)
                if done:
                    break
                state = next_state
                if state_keys is not None:
                    print_info(state_keys, state, action, reward)
    finally:
        # Note: to set all encoders and motor voltages to 0 and safely close the
        # board, you must call env.close()
        env.close()
    """


if __name__ == '__main__':
    # Enviroment vars
    state_keys = {
        'Aero-Position-v0': STATE_KEYS_AERO,
        'Qube-Inverted-Pendulum-v0': STATE_KEYS_QUBE,
        'Qube-Inverted-Pendulum-Sparse-v0': STATE_KEYS_QUBE
    }
    allowed_envs = list(state_keys.keys())

    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env',
        default='Aero-Position-v0',
        choices=allowed_envs,
        help='Enviroment to run.')
    parser.add_argument(
        '--control',
        default='random',
        choices=['random', 'none', 'classic'],
        help='Select what type of action to take.')
    args, _ = parser.parse_known_args()

    # Get action function
    controllers = {
        'none': NoControl,
        'random': RandomControl,
        'classic': AeroClassicControl if args.env == 'Aero-Position-v0' else QubeHoldInvetedClassicControl
    }

    print('Testing {}'.format(args.env))
    test_env(args.env, controllers[args.control], state_keys=state_keys[args.env])
