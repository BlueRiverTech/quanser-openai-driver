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


# No input
def zero_policy(state, **kwargs):
    return np.array([0.0])


# Constant input
def constant_policy(state, **kwargs):
    return np.array([3.0])


# Rand input
def random_policy(state, **kwargs):
    return np.asarray([np.random.randn()])


# Square wave, switch every 85 ms
def square_wave_policy(state, step, frequency=250, **kwargs):
    steps_until_85ms = int(85 * (frequency / 300))
    state = _convert_state(state)
    # Switch between positive and negative every 85 ms
    mod_170ms = step % (2 * steps_until_85ms)
    if mod_170ms < steps_until_85ms:
        action = 3.0
    else:
        action = -3.0
    return np.array([action])


# Flip policy
def energy_control_policy(state, **kwargs):
    state = _convert_state(state)
    # Run energy-based control to flip up the pendulum
    theta, alpha, theta_dot, alpha_dot = state
    # alpha_dot += alpha_dot + 1e-15

    """Implements a energy based swing-up controller"""
    mu = 50.0  # in m/s/J
    ref_energy = 30.0 / 1000.0  # Er in joules

    max_u = 6  # Max action is 6m/s^2

    # System parameters
    jp = 3.3282e-5
    lp = 0.129
    lr = 0.085
    mp = 0.024
    mr = 0.095
    rm = 8.4
    g = 9.81
    kt = 0.042

    pend_torque = (1 / 2) * mp * g * lp * (1 + np.cos(alpha))
    energy = pend_torque + (jp / 2.0) * alpha_dot * alpha_dot

    u = mu * (energy - ref_energy) * np.sign(-1 * np.cos(alpha) * alpha_dot)
    u = np.clip(u, -max_u, max_u)

    torque = (mr * lr) * u
    voltage = (rm / kt) * torque
    return np.array([-voltage])


# Hold policy
def pd_control_policy(state, **kwargs):
    state = _convert_state(state)
    theta, alpha, theta_dot, alpha_dot = state
    # multiply by proportional and derivative gains
    kp_theta = -2.0
    kp_alpha = 35.0
    kd_theta = -1.5
    kd_alpha = 3.0

    # If pendulum is within 20 degrees of upright, enable balance control, else zero
    if np.abs(alpha) <= (20.0 * np.pi / 180.0):
        action = (
            theta * kp_theta
            + alpha * kp_alpha
            + theta_dot * kd_theta
            + alpha_dot * kd_alpha
        )
    else:
        action = 0.0
    action = np.clip(action, -3.0, 3.0)
    return np.array([action])


# Flip and Hold
def flip_and_hold_policy(state, **kwargs):
    state = _convert_state(state)
    theta, alpha, theta_dot, alpha_dot = state

    # If pendulum is within 20 degrees of upright, enable balance control
    if np.abs(alpha) <= (20.0 * np.pi / 180.0):
        action = pd_control_policy(state)
    else:
        action = energy_control_policy(state)
    return action


# Square wave instead of energy controller flip and hold
def square_wave_flip_and_hold_policy(state, **kwargs):
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
        kp_theta = 0.9
        kp_alpha = -3.6
        kd_theta = 0.7
        kd_alpha = -0.285
        if alpha >= 0:
            action = (
                theta * kp_theta
                + (alpha - np.pi) * kp_alpha
                + theta_dot * kd_theta
                + alpha_dot * kd_alpha
            )
        else:
            action = (
                theta * kp_theta
                + (alpha + np.pi) * kp_alpha
                + theta_dot * kd_theta
                + alpha_dot * kd_alpha
            )
    else:
        action = 0

    action = np.clip(action, -3.0, 3.0)
    return np.array([action], dtype=np.float64)


# Hold policy
def pd_tracking_control_policy(state, **kwargs):
    state = _convert_state(state)
    theta, alpha, theta_dot, alpha_dot, theta_target = state
    # multiply by proportional and derivative gains
    kp_theta = -2.0
    kp_alpha = 35.0
    kd_theta = -1.5
    kd_alpha = 3.0

    # If pendulum is within 20 degrees of upright, enable balance control, else zero
    if np.abs(alpha) <= (20.0 * np.pi / 180.0):
        action = (
            (theta - theta_target) * kp_theta
            + alpha * kp_alpha
            + theta_dot * kd_theta
            + alpha_dot * kd_alpha
        )
    else:
        action = 0.0
    action = np.clip(action, -3.0, 3.0)
    return np.array([action])
