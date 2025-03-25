from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from scipy.integrate import odeint
import numpy as np
import math
from load_config import load_config

cfg = load_config()
# Motor
Rm = cfg['Rm']  # Resistance
kt = cfg['kt']  # Current-torque (N-m/A)
km = cfg['km']  # Back-emf constant (V-s/rad)

# Rotary Arm
mr = cfg['mr']  # Mass (kg)
Lr = cfg['Lr']  # Total length (m)
Jr = mr * Lr ** 2 / 12  # Moment of inertia about pivot (kg-m^2)
Dr = cfg['Dr']  # Equivalent viscous damping coefficient (N-m-s/rad)

# Pendulum Link
mp = cfg['mp']  # Mass (kg)
Lp = cfg['Lp']  # Total length (m)
Jp = mp * Lp ** 2 / 12  # Moment of inertia about pivot (kg-m^2)
Dp = cfg['Dp']  # Equivalent viscous damping coefficient (N-m-s/rad)

g = cfg['g']  # Gravity constant


def diff_forward_model_ode(state, t, action, dt):
    theta, alpha, theta_dot, alpha_dot = state
    Vm = action
    tau = -(km * (Vm - km * theta_dot)) / Rm  # torque

    # fmt: off
    # From Rotary Pendulum Workbook
    theta_dot_dot = (-Lp*Lr*mp*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha))*np.cos(alpha) + (4.0*Jp + Lp**2*mp)*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*np.sin(2.0*alpha) + 2.0*Lp*Lr*alpha_dot**2*mp*np.sin(alpha) - 4.0*tau))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp))
    alpha_dot_dot = (2.0*Lp*Lr*mp*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*np.sin(2.0*alpha) + 2.0*Lp*Lr*alpha_dot**2*mp*np.sin(alpha) - 4.0*tau)*np.cos(alpha) - 0.5*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp)*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha)))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp))
    # fmt: on

    diff_state = np.array([theta_dot, alpha_dot, theta_dot_dot, alpha_dot_dot]).reshape(
        (4,)
    )
    diff_state = np.array(diff_state, dtype="float64")
    return diff_state


def forward_model_ode(theta, alpha, theta_dot, alpha_dot, Vm, dt, integration_steps):
    t = np.linspace(0.0, dt, 2)  # TODO: add and check integration steps here

    state = np.array([theta, alpha, theta_dot, alpha_dot])
    next_state = np.array(odeint(diff_forward_model_ode, state, t, args=(Vm, dt)))[1, :]
    theta, alpha, theta_dot, alpha_dot = next_state

    theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
    alpha = ((alpha + np.pi) % (2 * np.pi)) - np.pi

    return theta, alpha, theta_dot, alpha_dot


def forward_model_euler(theta, alpha, theta_dot, alpha_dot, Vm, dt, integration_steps):
    dt /= integration_steps
    for step in range(integration_steps):
        tau = -(km * (Vm - km * theta_dot)) / Rm  # torque

        # fmt: off
        # From Rotary Pendulum Workbook
        theta_dot_dot = float((-Lp*Lr*mp*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha))*np.cos(alpha) + (4.0*Jp + Lp**2*mp)*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*np.sin(2.0*alpha) + 2.0*Lp*Lr*alpha_dot**2*mp*np.sin(alpha) - 4.0*tau))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp)))
        alpha_dot_dot = float((2.0*Lp*Lr*mp*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*np.sin(2.0*alpha) + 2.0*Lp*Lr*alpha_dot**2*mp*np.sin(alpha) - 4.0*tau)*np.cos(alpha) - 0.5*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp)*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha)))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp)))
        # fmt: on

        if True:  # semi-implicit euler (more accurate)
            theta_dot += theta_dot_dot * dt
            alpha_dot += alpha_dot_dot * dt
            theta += theta_dot * dt
            alpha += alpha_dot * dt
        else:  # classic euler
            theta += theta_dot * dt
            alpha += alpha_dot * dt
            theta_dot += theta_dot_dot * dt
            alpha_dot += alpha_dot_dot * dt

        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        alpha = ((alpha + np.pi) % (2 * np.pi)) - np.pi

    return theta, alpha, theta_dot, alpha_dot
