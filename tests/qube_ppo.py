from __future__ import absolute_import, print_function, division

import tensorflow as tf
import numpy as np
import argparse
import time
import gym
import os

from gym_brt.envs import QubeBeginUprightEnv

try:
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from baselines.ppo2.ppo2 import learn as learn_ppo2
except:
    raise ImportError('Please install OpenAI baselines from: https://github.com/openai/baselines')


def main(args):
    network = 'mlp'
    network_kwargs = {'num_layers':2, 'num_hidden':64, 'activation':tf.tanh}

    try:
        env = lambda *a, **k: QubeBeginUprightEnv(frequency=300)
        env = DummyVecEnv([env])

        model = learn_ppo2(
            network=network, env=env, total_timesteps=int(float(args.num_steps)),
            nsteps=2048, ent_coef=0.0, lr=lambda f: 3e-4 * f,
            vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=1, nminibatches=32, noptepochs=10, cliprange=0.2,
            load_path=args.load_path,
            **network_kwargs)

        if args.save_path is not None:
            model.save(args.save_path)

        if args.play:
            e = env.envs[0]
            obs = e.reset()
            while True:
                t = time.time()
                actions, _, state, _ = model.step(obs)
                print(actions)
                print('Time of step: ', time.time() - t)
                obs, _, done, _ = e.step(actions[0])
                done = done.any() if isinstance(done, np.ndarray) else done

                if done:
                    obs = e.reset()
    finally:
        for e in env.envs:
            e.close()


if __name__ == '__main__':

    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_steps', '-n',
        default=0,
        help='Total number of steps to run.')
    parser.add_argument(
        '--frequency', '-f',
        default='1000',
        type=float,
        help='The frequency of samples on the Quanser hardware.')
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--load_path', '-l', type=str)
    parser.add_argument('-p', '--play', action='store_true')
    args, _ = parser.parse_known_args()

    main(args)