# Quanser OpenAI Driver
Has an OpenAI Gym wrapper for the Quanser Qube Servo 2 and Quanser Aero


# Installation
This was _ONLY_ tested on Ubuntu 16.04 LTS and Ubunbtu 18.04 LTS using Python 2.7. <br>
To compile the Cython wrapper, you need a to have the Cython package installed (we recommend using virtualenv). <br>
Cython is not needed once you have the wrapper compiled (only Python). <br>

To compile and run a test
1. Install the HIL SDK
    - `cd <installation-directory>/quanser-openai-driver/hil_sdk_linux_x86_64 && ./setup_hil_sdk`
2. Compile the Cython wrapper
    - `cd <installation-directory>/quanser-openai-driver/gym-enviroments/gym_brt/envs/QuanserWrapper && make`
3. Install gym_qube
    - `cd <installation-directory>/quanser-openai-driver/gym-enviroments && pip install -e .`
4. Run the random agent test
    - For the Qube: `python <installation-directory>/quanser-openai-driver/tests/test_inverted_pendulum.py`
    - For the Aero: `python <installation-directory>/quanser-openai-driver/tests/test_aero.py`

The `hil_sdk_linux_x86_64` directory contains the installation files for the HIL SDK (Copyright (C) Quanser Inc.) <br>
The `gym-enviroments` directory contains a Python and OpenAI Gym wrapper around the HIL SDK


# Usage
Usage is very similar to most OpenAI gym enviroments but **requires** that you close the enviroment when finished.
Without safely closing the Env, bad things may happen. Usually you will not be able to reopen the board.

This can be done with a context manager using a `with` statement
```python
import gym
from gym_brt import QubeInvertedPendulumEnv

num_episodes = 10
num_steps = 250

with QubeInvertedPendulumEnv() as env:
    for episode in range(num_episodes):
        state = env.reset()
        for step in range(num_steps):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
```

Or can be closed manually by using `env.close()`
```python
import gym
from gym_brt import AeroPositionEnv

num_episodes = 10
num_steps = 250

env = AeroPositionEnv()
try:
    for episode in range(num_episodes):
        state = env.reset()
        for step in range(num_steps):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
finally:
    env.close()
```


# Warning
Forgetting to close the envrioment or incorrectly closing the env leads to several possible issues. The worst including segfaults.

The most common case when the env was not properly closed: you can not reopen the env and you get:
```
Error 1073: QERR_BOARD_ALREADY_OPEN: The HIL board is already opened by another process. The board does not support access from more than one process at the same time.
```
The easiest solution to this is to (note the order):
1. Unplug USB from the Qube/Aero
1. Unplug power from USB/Aero
1. Restart your computer
1. Replug power to Qube/Aero
1. Replug USB into the Qube/Aero

Once you restarted, you can try running the program again (with proper closing).


![Qube Standing Up](/QUBE-Servo_2_angled_pendulum.jpg?raw=true)



