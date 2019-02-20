from __future__ import absolute_import, print_function, division

import tensorflow as tf
import numpy as np
import argparse
import time
import gym

from gym_brt.envs import QubeBeginUprightEnv, QubeBeginDownEnv

try:
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from baselines.ppo2.ppo2 import learn as learn_ppo2
    from baselines import logger
except:
    raise ImportError('Please install OpenAI baselines from: https://github.com/openai/baselines')


def main(args):
    if args.network == 'mlp':
        network_kwargs = {'num_layers':2, 'num_hidden':64, 'activation':tf.tanh, 'layer_norm': False}
    elif args.network == 'lstm':
        network_kwargs = {'nlstm':128, 'layer_norm': False}
    else:
        raise ValueError('{} is not a valid network type.'.format(args.network))

    print('Using a {} network'.format(args.network))

    try:
        logger.configure()

        if args.env == 'up':
            qube_env = QubeBeginUprightEnv
        elif args.env == 'down':
            qube_env = QubeBeginDownEnv
        else:
            raise ValueError

        env = lambda *a, **k: qube_env(frequency=args.frequency)
        env = DummyVecEnv([env])

        model = learn_ppo2(
            network=args.network, env=env,
            total_timesteps=int(float(args.num_steps)),
            nsteps=2048, ent_coef=0.0, lr=lambda f: 3e-4 * f,
            vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=1, nminibatches=32, noptepochs=10, cliprange=0.2,
            save_interval=50000,
            load_path=args.load_path,
            **network_kwargs)

        if args.save_path is not None:
            model.save(args.save_path)

        if args.play > 0:
            e = env.envs[0]
            obs = e.reset()
            for _ in range(int(args.play * args.frequency)): # Run for `play` seconds
                t = time.time()
                actions, _, state, _ = model.step(obs)
                print('Time of step: ', time.time() - t)
                obs, r, done, _ = e.step(actions[0])
                print('Reward', r)

                print(obs, actions)
                done = done.any() if isinstance(done, np.ndarray) else done

                if done:
                    obs = e.reset()
            obs = e.reset()

    finally:
        for e in env.envs:
            e.close()


if __name__ == '__main__':

    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e',
        '--env',
        default='up',
        choices=['up', 'down'],
        help='Enviroment to run.')
    parser.add_argument(
        '--network',
        '--nn',
        default='mlp',
        choices=['mlp', 'lstm'],
        help='Type of neural network to use.')
    parser.add_argument(
        '--num_steps', '-n',
        default=0,
        help='Total number of steps to run.')
    parser.add_argument(
        '--frequency', '-f',
        default='300',
        type=float,
        help='The frequency of samples on the Quanser hardware.')
    parser.add_argument(
        '--play', '-p',
        default='0',
        type=float,
        help='Play trained model for set number of seconds.')
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--load_path', '-l', type=str)
    # parser.add_argument('--checkpoint_dir', '-c', type=str)
    args, _ = parser.parse_known_args()

    main(args)
