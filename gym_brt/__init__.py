from gym.envs.registration import register

### Hardware Environments ######################################################
# Swingup Environments
register(
    id='QubeSwingupEnv-v0',
    entry_point='gym_brt.envs:QubeSwingupEnv',
    nondeterministic=True,
    max_episode_steps=2048,
    kwargs={'use_simulator': False}
)
register(
    id='QubeSwingupSparseEnv-v0',
    entry_point='gym_brt.envs:QubeSwingupSparseEnv',
    nondeterministic=True,
    max_episode_steps=2048,
    kwargs={'use_simulator': False}
)
register(
    id='QubeSwingupFollowEnv-v0',
    entry_point='gym_brt.envs:QubeSwingupFollowEnv',
    nondeterministic=True,
    max_episode_steps=2048,
    kwargs={'use_simulator': False}
)
register(
    id='QubeSwingupFollowSparseEnv-v0',
    entry_point='gym_brt.envs:QubeSwingupFollowSparseEnv',
    nondeterministic=True,
    max_episode_steps=2048,
    kwargs={'use_simulator': False}
)
# Balance Environments
register(
    id='QubeBalanceEnv-v0',
    entry_point='gym_brt.envs:QubeBalanceEnv',
    nondeterministic=True,
    max_episode_steps=2048,
    kwargs={'use_simulator': False}
)
register(
    id='QubeBalanceSparseEnv-v0',
    entry_point='gym_brt.envs:QubeBalanceSparseEnv',
    nondeterministic=True,
    max_episode_steps=2048,
    kwargs={'use_simulator': False}
)
register(
    id='QubeBalanceFollowEnv-v0',
    entry_point='gym_brt.envs:QubeBalanceFollowEnv',
    nondeterministic=True,
    max_episode_steps=2048,
    kwargs={'use_simulator': False}
)
register(
    id='QubeBalanceFollowSparseEnv-v0',
    entry_point='gym_brt.envs:QubeBalanceFollowSparseEnv',
    nondeterministic=True,
    max_episode_steps=2048,
    kwargs={'use_simulator': False}
)
# Dampen Environments
register(
    id='QubeDampenEnv-v0',
    entry_point='gym_brt.envs:QubeDampenEnv',
    nondeterministic=True,
    max_episode_steps=2048,
    kwargs={'use_simulator': False}
)
register(
    id='QubeDampenSparseEnv-v0',
    entry_point='gym_brt.envs:QubeDampenSparseEnv',
    nondeterministic=True,
    max_episode_steps=2048,
    kwargs={'use_simulator': False}
)
register(
    id='QubeDampenFollowEnv-v0',
    entry_point='gym_brt.envs:QubeDampenFollowEnv',
    nondeterministic=True,
    max_episode_steps=2048,
    kwargs={'use_simulator': False}
)
register(
    id='QubeDampenFollowSparseEnv-v0',
    entry_point='gym_brt.envs:QubeDampenFollowSparseEnv',
    nondeterministic=True,
    max_episode_steps=2048,
    kwargs={'use_simulator': False}
)
register(
    id='QubeBalanceFollowSineWaveEnv-v0',
    entry_point='gym_brt.envs:QubeBalanceFollowSineWaveEnv',
    nondeterministic=True,
    max_episode_steps=2048,
    kwargs={'use_simulator': False}
)
# Sin wave environments
register(
    id='QubeSwingupFollowSineWaveEnv-v0',
    entry_point='gym_brt.envs:QubeSwingupFollowSineWaveEnv',
    nondeterministic=True,
    max_episode_steps=2048,
    kwargs={'use_simulator': False}
)
register(
    id='QubeRotorFollowSineWaveEnv-v0',
    entry_point='gym_brt.envs:QubeRotorFollowSineWaveEnv',
    nondeterministic=True,
    max_episode_steps=2048,
    kwargs={'use_simulator': False}
)
register(
    id='QubeDampenFollowSineWaveEnv-v0',
    entry_point='gym_brt.envs:QubeDampenFollowSineWaveEnv',
    nondeterministic=True,
    max_episode_steps=2048,
    kwargs={'use_simulator': False}
)
# Rotor Evironments
# Note: the rotor environments have issues with the rewards.
# register(
#     id='QubeRotorEnv-v0',
#     entry_point='gym_brt.envs:QubeRotorEnv',
#     nondeterministic=True,
#     max_episode_steps=2048,
#     kwargs={'use_simulator': False}
# )
# register(
#     id='QubeRotorFollowEnv-v0',
#     entry_point='gym_brt.envs:QubeRotorFollowEnv',
#     nondeterministic=True,
#     max_episode_steps=2048,
#     kwargs={'use_simulator': False}
# )

### Simulated Environments #####################################################
# Swingup Environments
register(
    id='QubeSwingupSimEnv-v0',
    entry_point='gym_brt.envs:QubeSwingupEnv',
    max_episode_steps=2048,
    kwargs={'use_simulator': True}
)
register(
    id='QubeSwingupSparseSimEnv-v0',
    entry_point='gym_brt.envs:QubeSwingupSparseEnv',
    max_episode_steps=2048,
    kwargs={'use_simulator': True}
)
register(
    id='QubeSwingupFollowSimEnv-v0',
    entry_point='gym_brt.envs:QubeSwingupFollowEnv',
    max_episode_steps=2048,
    kwargs={'use_simulator': True}
)
register(
    id='QubeSwingupFollowSparseSimEnv-v0',
    entry_point='gym_brt.envs:QubeSwingupFollowSparseEnv',
    max_episode_steps=2048,
    kwargs={'use_simulator': True}
)
# Balance Environments
register(
    id='QubeBalanceSimEnv-v0',
    entry_point='gym_brt.envs:QubeBalanceEnv',
    max_episode_steps=2048,
    kwargs={'use_simulator': True}
)
register(
    id='QubeBalanceSparseSimEnv-v0',
    entry_point='gym_brt.envs:QubeBalanceSparseEnv',
    max_episode_steps=2048,
    kwargs={'use_simulator': True}
)
register(
    id='QubeBalanceFollowSimEnv-v0',
    entry_point='gym_brt.envs:QubeBalanceFollowEnv',
    max_episode_steps=2048,
    kwargs={'use_simulator': True}
)
register(
    id='QubeBalanceFollowSparseSimEnv-v0',
    entry_point='gym_brt.envs:QubeBalanceFollowSparseEnv',
    max_episode_steps=2048,
    kwargs={'use_simulator': True}
)
# Dampen Environments
register(
    id='QubeDampenSimEnv-v0',
    entry_point='gym_brt.envs:QubeDampenEnv',
    max_episode_steps=2048,
    kwargs={'use_simulator': True}
)
register(
    id='QubeDampenSparseSimEnv-v0',
    entry_point='gym_brt.envs:QubeDampenSparseEnv',
    max_episode_steps=2048,
    kwargs={'use_simulator': True}
)
register(
    id='QubeDampenFollowSimEnv-v0',
    entry_point='gym_brt.envs:QubeDampenFollowEnv',
    max_episode_steps=2048,
    kwargs={'use_simulator': True}
)
register(
    id='QubeDampenFollowSparseSimEnv-v0',
    entry_point='gym_brt.envs:QubeDampenFollowSparseEnv',
    max_episode_steps=2048,
    kwargs={'use_simulator': True}
)
register(
    id='QubeBalanceFollowSineWaveSimEnv-v0',
    entry_point='gym_brt.envs:QubeBalanceFollowSineWaveEnv',
    max_episode_steps=2048,
    kwargs={'use_simulator': True}
)
# Sin wave environments
register(
    id='QubeSwingupFollowSineWaveSimEnv-v0',
    entry_point='gym_brt.envs:QubeSwingupFollowSineWaveEnv',
    max_episode_steps=2048,
    kwargs={'use_simulator': True}
)
register(
    id='QubeRotorFollowSineWaveSimEnv-v0',
    entry_point='gym_brt.envs:QubeRotorFollowSineWaveEnv',
    max_episode_steps=2048,
    kwargs={'use_simulator': True}
)
register(
    id='QubeDampenFollowSineWaveSimEnv-v0',
    entry_point='gym_brt.envs:QubeDampenFollowSineWaveEnv',
    max_episode_steps=2048,
    kwargs={'use_simulator': True}
)
# Rotor Evironments
# Note: the rotor environments have issues with the rewards.
# register(
#     id='QubeRotorSimEnv-v0',
#     entry_point='gym_brt.envs:QubeRotorEnv',
#     max_episode_steps=2048,
#     kwargs={'use_simulator': True}
# )
# register(
#     id='QubeRotorFollowSimEnv-v0',
#     entry_point='gym_brt.envs:QubeRotorFollowEnv',
#     max_episode_steps=2048,
#     kwargs={'use_simulator': True}
# )


