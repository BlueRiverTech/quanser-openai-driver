# Quanser OpenAI Driver
Has an OpenAI Gym wrapper for the Quanser Qube Servo 2 and Quanser Aero

- [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Recompiling Cython](#recompiling-cython-code)
- [Basic Usage](#usage)
- [Warning](#warning)


# Setup
We have tested on Ubuntu 16.04 LTS and Ubuntu 18.04 LTS using Python 2.7 and Python 3.6.5<br>


### Prerequisites
Install the HIL SDK from Quanser.<br>
A mirror is available at https://github.com/quanser/hil_sdk_linux_x86_64.<br>

You can install the driver by:
```bash
    git clone https://github.com/quanser/hil_sdk_linux_x86_64.git
    sudo chmod a+x ./hil_sdk_linux_x86_64/setup_hil_sdk ./hil_sdk_linux_x86_64/uninstall_hil_sdk
    sudo ./hil_sdk_linux_x86_64/setup_hil_sdk
```

You also must have pip installed:
```bash
    sudo apt-get install python3-pip
```


### Installation
We recommend that you use a virtual environment such as [conda (recommended)](https://conda.io/docs/user-guide/getting-started.html), [virtualenv](https://virtualenv.pypa.io/en/stable/), or [Pipenv](https://pipenv.readthedocs.io/en/latest/)

You can install the driver by cloning and pip-installing:
```bash
    git clone https://github.com/BlueRiverTech/quanser-openai-driver.git
    cd quanser-openai-driver
    pip3 install -e .
```

Once you have that setup: Run the classical control baseline (ensure the Qube is connected to your computer)<br>
```bash
python tests/test.py --env QubeSwingupEnv --controller flip
```


# Usage
Usage is very similar to most OpenAI gym environments but **requires** that you close the environment when finished.
Without safely closing the Env, bad things may happen. Usually you will not be able to reopen the board.

This can be done with a context manager using a `with` statement
```python
import gym
from gym_brt import QubeSwingupEnv

num_episodes = 10
num_steps = 250

with QubeSwingupEnv() as env:
    for episode in range(num_episodes):
        state = env.reset()
        for step in range(num_steps):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
```

Or can be closed manually by using `env.close()`. You can see an [example here](docs/alternatives.md#usage).


# Environments
Information about various environments can be found in [docs/envs](docs/envs.md) and our [whitepaper](https://arxiv.org/abs/2001.02254).


# Control
Information about baselines can be found in [docs/control](docs/control.md).


# Hardware Wrapper
Information about the Python wrapper for Quanser hardware and Qube Servo 2 simulator can be found in [docs/quanser](docs/quanser.md) and our [whitepaper](https://arxiv.org/abs/2001.02254).


# Citing
If you use this in your research please cite the following [whitepaper](https://arxiv.org/abs/2001.02254):

```
@misc{2001.02254,
  author = {{Polzounov}, Kirill and {Sundar}, Ramitha and {Redden}, Lee},
  title = "{Blue River Controls: A toolkit for Reinforcement Learning Control Systems on Hardware}",
  year = {2019},
  eprint = {arXiv:2001.02254},
  howpublished = {Accepted at the Workshop on Deep Reinforcement Learning at the 33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.}
}
```
