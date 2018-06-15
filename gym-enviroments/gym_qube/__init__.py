from gym.envs.registration import register

register(
    id='Qube-v0',
    entry_point='gym_qube.envs:QubeEnv',
)
