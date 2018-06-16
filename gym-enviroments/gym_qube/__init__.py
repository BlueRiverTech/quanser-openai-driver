from gym.envs.registration import register

register(
    id='Qube-Inverted-Pendulum-v0',
    entry_point='gym_qube.envs:QubeInvertedPendulumEnv',
)

register(
    id='Qube-Inverted-Pendulum-Sparse-v0',
    entry_point='gym_qube.envs:QubeInvertedPendulumSparseRewardEnv',
)
