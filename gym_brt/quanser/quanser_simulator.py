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
Jr = mr * Lr ** 2 / 12  # Moment of inertia about pivot (kg-m^2)
Dr = 0.0015  # Equivalent viscous damping coefficient (N-m-s/rad)

# Pendulum Link
mp = 0.024  # Mass (kg)
Lp = 0.129  # Total length (m)
Jp = mp * Lp ** 2 / 12  # Moment of inertia about pivot (kg-m^2)
Dp = 0.0005  # Equivalent viscous damping coefficient (N-m-s/rad)

g = 9.81  # Gravity constant


def diff_forward_model_ode(state, t, action, dt):
    theta, alpha, theta_dot, alpha_dot = state
    Vm = action
    tau = (km * (Vm - km * theta_dot)) / Rm  # torque

    # fmt: off
    # From Rotary Pendulum Workbook (with change of var to make alpha = alpha + pi)
    theta_dot_dot = (-Lp*Lr*mp*(8.0*Dp*alpha_dot - Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha))*np.cos(alpha) + (4.0*Jp + Lp**2*mp)*(4.0*Dr*theta_dot + Lp**2*mp*alpha_dot*theta_dot*np.sin(2.0*alpha) - 2.0*Lp*Lr*mp*alpha_dot**2*np.sin(alpha) - 4.0*tau))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp))
    alpha_dot_dot = (-2.0*Lp*Lr*mp*(4.0*Dr*theta_dot + Lp**2*mp*alpha_dot*theta_dot*np.sin(2.0*alpha) - 2.0*Lp*Lr*mp*alpha_dot**2*np.sin(alpha) - 4.0*tau)*np.cos(alpha) + (2.0*Jr + 0.5*Lp**2*mp*np.sin(alpha)**2 + 2.0*Lr**2*mp)*(8.0*Dp*alpha_dot - Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha)))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp))
    # fmt: on

    diff_state = np.array([theta_dot, alpha_dot, theta_dot_dot, alpha_dot_dot]).reshape(
        (4,)
    )
    diff_state = np.array(diff_state, dtype="float64")
    return diff_state


def forward_model_ode(
    theta, alpha, alpha_unorm, theta_dot, alpha_dot, Vm, dt, integration_steps
):
    t = np.linspace(0.0, dt, 2)  # TODO: add and check integration steps here

    state = np.array([theta, alpha, theta_dot, alpha_dot])
    next_state = np.array(odeint(diff_forward_model_ode, state, t, args=(Vm, dt)))[1, :]
    theta, alpha, theta_dot, alpha_dot = state

    theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
    alpha = ((alpha + np.pi) % (2 * np.pi)) - np.pi

    # Ensure that the alpha encoder value is the same as in the Qube
    alpha_unnorm += alpha_dot * dt

    return theta, alpha, alpha_unnorm, theta_dot, alpha_dot


def forward_model_euler(
    theta, alpha, alpha_unorm, theta_dot, alpha_dot, Vm, dt, integration_steps
):
    dt /= integration_steps
    for step in range(integration_steps):
        tau = (km * (Vm - km * theta_dot)) / Rm  # torque

        # fmt: off
        # From Rotary Pendulum Workbook (with change of var to make alpha = alpha + pi)
        theta_dot_dot = float((-Lp*Lr*mp*(8.0*Dp*alpha_dot - Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha))*np.cos(alpha) + (4.0*Jp + Lp**2*mp)*(4.0*Dr*theta_dot + Lp**2*mp*alpha_dot*theta_dot*np.sin(2.0*alpha) - 2.0*Lp*Lr*mp*alpha_dot**2*np.sin(alpha) - 4.0*tau))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp)))
        alpha_dot_dot = float((-2.0*Lp*Lr*mp*(4.0*Dr*theta_dot + Lp**2*mp*alpha_dot*theta_dot*np.sin(2.0*alpha) - 2.0*Lp*Lr*mp*alpha_dot**2*np.sin(alpha) - 4.0*tau)*np.cos(alpha) + (2.0*Jr + 0.5*Lp**2*mp*np.sin(alpha)**2 + 2.0*Lr**2*mp)*(8.0*Dp*alpha_dot - Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha)))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp)))
        # fmt: on

        theta_dot += theta_dot_dot * dt
        alpha_dot += alpha_dot_dot * dt

        theta += theta_dot * dt
        alpha += alpha_dot * dt

        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        alpha = ((alpha + np.pi) % (2 * np.pi)) - np.pi

    return theta, alpha, alpha_unorm, theta_dot, alpha_dot


# Use numba if installed
try:
    import numba

    diff_forward_model_ode = numba.jit(diff_forward_model_ode)
    forward_model_ode = numba.jit(forward_model_ode)
    forward_model_euler = numba.jit(forward_model_euler)
except:
    print("Warning: Install 'numba' for faster simulation.")


class QubeServo2Simulator(object):
    """Simulator that has the same interface as the hardware wrapper."""

    def __init__(
        self,
        forward_model=None,
        safe_operating_voltage=18.0,
        integration_steps=1,
        frequency=1000,
    ):
        self._dt = 1.0 / frequency
        self._integration_steps = integration_steps
        self._max_voltage = safe_operating_voltage
        self.state = np.array([0, 0, 0, 0, 0]) + np.random.randn(5) * 0.01

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def reset_up(self):
        self.state = np.array([0, 0, 0, 0, 0]) + np.random.randn(5) * 0.01

    def reset_down(self):
        self.state = np.array([0, np.pi, np.pi, 0, 0]) + np.random.randn(5) * 0.01

    def reset_encoders(self):
        pass

    def action(self, action):
        action = np.clip(action, -self._max_voltage, self._max_voltage)
        self.state = forward_model(
            *self.state, action, self._dt, self._integration_steps
        )

        # Convert the angles into encoder counts
        # TODO: test if either of these should be negated
        theta_counts = int(self.state[0] / (np.pi / 1024))
        alpha_counts = int(self.state[2] / (np.pi / 1024))  # use unnormalized alpha

        encoders = [theta_counts, alpha_counts]
        currents = [action / 8.4]  # resistance is 8.4 ohms
        others = [0.0]  # [tach0, other_stuff]

        return currents, encoders, others


class QubeServo2SimulatorEuler(QubeServo2Simulator):
    def __init__(self, **kwargs):
        super(QubeServo2SimulatorEuler, self).__init__(
            forward_model=forward_model_euler, **kwargs
        )


class QubeServo2SimulatorODE(QubeServo2Simulator):
    def __init__(self, **kwargs):
        super(QubeServo2SimulatorODE, self).__init__(
            forward_model=forward_model_ode, **kwargs
        )
