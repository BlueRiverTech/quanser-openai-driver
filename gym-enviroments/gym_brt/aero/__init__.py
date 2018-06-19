from gym.envs.registration import register

register(
    id='Aero-Position-v0',
    entry_point='gym_brt.aero.envs:AeroPositionEnv',
)
