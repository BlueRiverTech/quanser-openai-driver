from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import time
import argparse
import numpy as np

from gym_brt import \
    AeroPositionEnv, \
    QubeFlipUpEnv, \
    QubeHoldInvertedEnv
from gym_brt.control import \
        NoControl, \
        RandomControl, \
        AeroClassicControl, \
        QubeHoldInvertedClassicControl, \
        QubeFlipUpInvertedClassicControl


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
             sample_freq=1000,
             state_keys=None,
             verbose=False):

    with env_name(sample_freq) as env:
        ctrl_sys = controller(env, sample_freq=sample_freq)
        for episode in range(num_episodes):
            state = env.reset()
            for step in range(num_steps):
                action = ctrl_sys.action(state)
                state, reward, done, _ = env.step(action)
                if done:
                    break
                if verbose and state_keys is not None:
                    print_info(state_keys, state, action, reward)

    """
    # Another way to run the Qube enviroment
    env = gym.make(env_name)
    try:
        for episode in range(num_episodes):
            state = env.reset()
            for step in range(num_steps):
                action = action_func(state)
                state, reward, done, _ = env.step(action)
                if done:
                    break
                if state_keys is not None:
                    print_info(state_keys, state, action, reward)
    finally:
        # Note: to set all encoders and motor voltages to 0 and safely close the
        # board, you must call env.close()
        env.close()
    """


def main():
    state_keys = {
        'AeroPositionEnv': STATE_KEYS_AERO,
        'QubeFlipUpEnv': STATE_KEYS_QUBE,
        'QubeHoldInvertedEnv': STATE_KEYS_QUBE
    }
    envs = {
        'AeroPositionEnv': AeroPositionEnv,
        'QubeFlipUpEnv': QubeFlipUpEnv,
        'QubeHoldInvertedEnv': \
            QubeHoldInvertedEnv
    }
    controllers = {
        'none': NoControl,
        'random': RandomControl,
        'classic': QubeFlipUpInvertedClassicControl,
        'qube-classic': QubeFlipUpInvertedClassicControl,
        'aero-classic': AeroClassicControl,
        'flip-up': QubeFlipUpInvertedClassicControl,
        'flip': QubeFlipUpInvertedClassicControl,
        'hold': QubeHoldInvertedClassicControl,
    }

    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e',
        '--env',
        default='QubeFlipUpEnv',
        choices=list(state_keys.keys()),
        help='Enviroment to run.')
    parser.add_argument(
        '-c',
        '--control',
        default='random',
        choices=list(controllers.keys()),
        help='Select what type of action to take.')
    parser.add_argument(
        '--num-episodes',
        default='10',
        type=int,
        help='Number of episodes to run.')
    parser.add_argument(
        '--num-steps',
        default='10000',
        type=int,
        help='Number of step to run per episode.')
    parser.add_argument(
        '-f',
        '--frequency',
        '--sample-frequency',
        default='1000',
        type=float,
        help='The frequency of samples on the Quanser hardware.')
    parser.add_argument('-v', '--verbose', action='store_true')
    args, _ = parser.parse_known_args()

    print('Testing Env:  {}'.format(args.env))
    print('Controller:   {}'.format(args.control))
    print('{} steps over {} episodes'.format(args.num_steps, args.num_episodes))
    print('Samples freq: {}'.format(args.frequency))
    test_env(
        envs[args.env],
        controllers[args.control],
        num_episodes=args.num_episodes,
        num_steps=args.num_steps,
        sample_freq=args.frequency,
        state_keys=state_keys[args.env],
        verbose=args.verbose)


if __name__ == '__main__':
    main()
