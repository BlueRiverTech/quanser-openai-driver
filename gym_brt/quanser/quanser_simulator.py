from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import math


# Motor
Rm = 8.4  # Resistance
kt = 0.042  # Current-torque (N-m/A)
km = 0.042  # Back-emf constant (V-s/rad)

# Rotary Arm
mr = 0.095  # Mass (kg)
Lr = 0.085  # Total length (m)
Jr = mr * Lr**2 / 12  # Moment of inertia about pivot (kg-m^2)
Dr = 0.0015  # Equivalent viscous damping coefficient (N-m-s/rad)

# Pendulum Link
mp = 0.024  # Mass (kg)
Lp = 0.129  # Total length (m)
Jp = mp * Lp**2 / 12  # Moment of inertia about pivot (kg-m^2)
Dp = 0.0005  # Equivalent viscous damping coefficient (N-m-s/rad)

g = 9.81  # Gravity constant


def forward_model(theta, alpha, theta_dot, alpha_dot, Vm, dt, euler_steps):

    dt /= euler_steps
    for step in range(euler_steps):
        tau = (km * (Vm - km * theta_dot)) / Rm  # torque

        theta_dot_dot = -(
            Lp * Lr * mp *
            (8.0 * Dp * alpha_dot - Lp**2 * mp * theta_dot**2 * math.sin(
                2.0 * alpha) + 4.0 * Lp * g * mp * math.sin(alpha)) *
            math.cos(alpha) + (4.0 * Jp + Lp**2 * mp) *
            (4.0 * Dr * theta_dot +
             Lp**2 * alpha_dot * mp * theta_dot * math.sin(2.0 * alpha) +
             2.0 * Lp * Lr * alpha_dot**2 * mp * math.sin(alpha) - 4.0 * tau)) / (
                 4.0 * Lp**2 * Lr**2 * mp**2 * math.cos(alpha)**2 +
                 (4.0 * Jp + Lp**2 * mp) *
                 (4.0 * Jr + Lp**2 * mp * math.sin(alpha)**2 + 4.0 * Lr**2 * mp))
        alpha_dot_dot = (
            4.0 * Lp * Lr * mp *
            (2.0 * Dr * theta_dot + 0.5 * Lp**2 * alpha_dot * mp * theta_dot *
             math.sin(2.0 * alpha) + Lp * Lr * alpha_dot**2 * mp * math.sin(alpha)
             - 2.0 * tau) * math.cos(alpha) -
            (4.0 * Jr + Lp**2 * mp * math.sin(alpha)**2 + 4.0 * Lr**2 * mp) *
            (4.0 * Dp * alpha_dot - 0.5 * Lp**2 * mp * theta_dot**2 *
             math.sin(2.0 * alpha) + 2.0 * Lp * g * mp * math.sin(alpha))) / (
                 4.0 * Lp**2 * Lr**2 * mp**2 * math.cos(alpha)**2 +
                 (4.0 * Jp + Lp**2 * mp) *
                 (4.0 * Jr + Lp**2 * mp * math.sin(alpha)**2 + 4.0 * Lr**2 * mp))

        theta_dot += theta_dot_dot * dt
        alpha_dot += alpha_dot_dot * dt

        theta += theta_dot * dt
        alpha += alpha_dot * dt

        theta %= (2 * math.pi)
        alpha %= (2 * math.pi)

    return theta, alpha, theta_dot, alpha_dot


# Use numba if installed
try:
    import numba
    forward_model = numba.jit(forward_model)
except:
    print('Warning: Install \'numba\' for faster simulation.')


class QubeServo2Simulator(object):
    '''Simulator that has the same interface as the hardware wrapper.'''
    def __init__(self,
                 safe_operating_voltage=18.0,
                 euler_steps=1,
                 frequency=1000):
        self._time_step = 1.0 / frequency
        self._euler_steps = euler_steps
        self.state = [0, 0, 0, 0]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def reset_encoders(self):
        pass
 
    def action(self, action):
        self.state = forward_model(
            *self.state,
            action,
            self._time_step,
            self._euler_steps)
        encoders = self.state[:2]  # [theta, alpha]
        currents = [action / 8.4]  # 8.4 is resistance
        others = [0.] #[tach0, other_stuff]

        return currents, encoders, others

