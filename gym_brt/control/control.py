from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def _convert_state(state):
    state = np.asarray(state)
    if state.shape == (4,):
        return state
    if state.shape == (5,):
        return state
    elif state.shape == (6,):
        # Get the angles
        theta_x, theta_y = state[0], state[1]
        alpha_x, alpha_y = state[2], state[3]
        theta = np.arctan2(theta_y, theta_x)
        alpha = np.arctan2(alpha_y, alpha_x)
        theta_dot, alpha_dot = state[4], state[5]
        return theta, alpha, theta_dot, alpha_dot
    else:
        raise ValueError(
            "State shape was expected to be one of: (4,1) or (6,1), found: {}".format(
                state.shape
            )
        )


def zero_policy(state, **kwargs):
    """Output a constant 0V action."""
    return np.array([0.0])


def constant_policy(state, **kwargs):
    """Output a constant 3V action."""
    return np.array([3.0])


def random_policy(state, **kwargs):
    """Output a random action sampled from a Gaussian"""
    return np.asarray([np.random.randn()])


def square_wave_policy(state, step, frequency=250, **kwargs):
    """Output a square wave with an amplitude of 3V and period of ~566ms."""
    state = _convert_state(state)
    steps_until_283ms = int(283 * (frequency / 1000))

    # Switch between positive and negative every 283ms
    mod_566ms = step % (2 * steps_until_283ms)
    if mod_566ms < steps_until_283ms:
        action = 3.0
    else:
        action = -3.0
    return np.array([action])


def energy_control_policy(state, **kwargs):
    """Uses an energy controller to flip the pendulum.

    Increase the energy of the pendulum until the total energy (sum of potential
    and kinetic) is equal to the reference energy Er, where Er is the maximum 
    potential energy (ie energy of the pendulum inverted and stationary)
    """
    state = _convert_state(state)
    theta, alpha, theta_dot, alpha_dot = state
    alpha_dot += alpha_dot + 1e-15
    mu = 50.0  # in m/s/J
    ref_energy = 30.0 / 1000.0  # Er in joules

    # System parameters
    lp, lr, mp, mr = 0.129, 0.085, 0.024, 0.095
    jp, rm, g, kt = 3.3282e-5, 8.4, 9.81, 0.042

    pend_torque = (1 / 2) * mp * g * lp * (1 + np.cos(alpha))
    energy = pend_torque + (jp / 2.0) * alpha_dot * alpha_dot
    u = mu * (energy - ref_energy) * np.sign(-1 * np.cos(alpha) * alpha_dot)
    torque = (mr * lr) * u
    voltage = (rm / kt) * torque
    return np.array([-voltage])


def pd_control_policy(state, ref_angle=0, **kwargs):
    """Use a PD controller to stabilize the inverted pendulum."""
    state = _convert_state(state)
    theta, alpha, theta_dot, alpha_dot = state
    # If pendulum is within 20 degrees of upright, enable balance control, else zero
    if np.abs(alpha) <= (20.0 * np.pi / 180.0):
        action = (
            (theta - ref_angle) * kp_theta
            + alpha * kp_alpha
            + theta_dot * kd_theta
            + alpha_dot * kd_alpha
        )
    else:
        action = 0.0
    return np.array([action])


def flip_and_hold_policy(state, **kwargs):
    """A complete mixed-mode controller that combines energy and PD control."""
    state = _convert_state(state)
    theta, alpha, theta_dot, alpha_dot = state

    # If pendulum is within 20 degrees of upright, enable balance control
    if np.abs(alpha) <= (20.0 * np.pi / 180.0):
        action = pd_control_policy(state)
    else:
        action = energy_control_policy(state)
    return action


def square_wave_flip_and_hold_policy(state, **kwargs):
    """Square wave instead of energy controller flip and hold """
    state = _convert_state(state)
    theta, alpha, theta_dot, alpha_dot = state

    # If pendulum is within 20 degrees of upright, enable balance control
    if np.abs(alpha) <= (20.0 * np.pi / 180.0):
        action = pd_control_policy(state)
    else:
        action = square_wave_policy(state, **kwargs)
    return action


def dampen_policy(state, **kwargs):
    state = _convert_state(state)
    theta, alpha, theta_dot, alpha_dot = state

    # Alt alpha angle: -pi to +pi, where 0 is the pendulum facing down (at rest)
    alt_alpha = (alpha + 2 * np.pi) % (2 * np.pi) - np.pi
    if np.abs(alt_alpha) < (20.0 * np.pi / 180.0) and np.abs(theta) < (np.pi / 4):
        kp_theta, kp_alpha, kd_theta, kd_alpha = -2.0, 35.0, -1.5, 3.0
        if alpha >= 0:
            action = (
                -theta * kp_theta
                + (np.pi - alpha) * kp_alpha
                + -theta_dot * kd_theta
                + -alpha_dot * kd_alpha
            )
        else:
            action = (
                -theta * kp_theta
                + (-np.pi - alpha) * kp_alpha
                + -theta_dot * kd_theta
                + -alpha_dot * kd_alpha
            )
    else:
        action = 0

    return np.array([action], dtype=np.float64)


def pd_tracking_control_policy(state, **kwargs):
    state = _convert_state(state)
    _, _, _, _, theta_target = state
    return pd_control_policy(state[:4], theta_target)
