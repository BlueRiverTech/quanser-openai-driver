from gym_brt.envs import AeroPositionEnv
from gym_brt.envs import QubeInvertedPendulumEnv
from gym_brt.envs import QubeInvertedPendulumSparseRewardEnv

from gym.envs.registration import register

register(
    id='Aero-Position-v0',
    entry_point='gym_brt.envs:AeroPositionEnv',
)

register(
    id='Qube-Inverted-Pendulum-v0',
    entry_point='gym_brt.qube.envs:QubeInvertedPendulumEnv',
)

register(
    id='Qube-Inverted-Pendulum-Sparse-v0',
    entry_point='gym_brt.qube.envs:QubeInvertedPendulumSparseRewardEnv',
)
