from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


from gym_brt.envs.QuanserSimulator.rotary_pendulum import \
    RotaryPendulumNonLinearApproximation, RotaryPendulumLinearApproximation

import pyximport; pyximport.install()
from gym_brt.envs.QuanserSimulator.c_rotary_pedulum import \
    CythonRotaryPendulumNonLinearApproximation


pendulums = {
    'RotaryPendulumNonLinearApproximation': RotaryPendulumNonLinearApproximation,
    'RotaryPendulumLinearApproximation': RotaryPendulumLinearApproximation,
    'CythonRotaryPendulumNonLinearApproximation': CythonRotaryPendulumNonLinearApproximation,
}

class QuanserSimulator(object):
    def __init__(self, pendulum='RotaryPendulumNonLinearApproximation',
                 safe_operating_voltage=18.0, euler_steps=1, frequency=1000):
        # Pendulum simulation
        p = pendulums[pendulum]
        def pendulum(state, action, time_step=1/frequency, euler_steps=euler_steps):
            p(state, action, time_step=time_step, euler_steps=euler_steps)
        self.pendulum = pendulum

        # Inital state
        self.state = np.array([0., 0., 0., 0.], dtype=np.float64)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass
 
    def action(self, action):
        self.state = self.pendulum(self.state, action)
        theta, alpha = self.state[:2]
        encoders = [theta, alpha]
        currents = [action / 8.4]  # 8.4 is resistance
        others = [0.] #[tach0, other_stuff]

        return currents, encoders, others
