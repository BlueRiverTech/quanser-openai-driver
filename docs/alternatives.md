# Alternatives

### Usage
In addition to a [context manager (recommended)](../README.md#usage), the environment can also be closed manually by using `env.close()`

```python
import gym
from gym_brt import QubeSwingupEnv

num_episodes = 10
num_steps = 250

env = QubeSwingupEnv()
try:
    for episode in range(num_episodes):
        state = env.reset()
        for step in range(num_steps):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
finally:
    env.close()
```
