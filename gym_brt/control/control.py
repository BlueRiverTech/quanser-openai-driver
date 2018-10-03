from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


# Set the motor saturation limits for the Aero and Qube
AERO_MAX_VOLTAGE = 15.0
QUBE_MAX_VOLTAGE = 6.0


class Control(object):
    def __init__(self, env=None, action_shape=None, *args, **kwargs):
        if env:
            self.action_shape = env.action_space.sample().shape
        elif action_shape:
            self.action_shape = action_shape
        else:
            raise ValueError('Either env or action_shape must be passed.')

    def action(self, state):
        raise NotImplementedError


class NoControl(Control):
    '''Output motor voltages of 0'''
    def __init__(self, env, *args, **kwargs):
        super(NoControl, self).__init__(env)
        self._action_space = env.action_space

    def action(self, state):
        return 0. * self._action_space.sample()


class RandomControl(Control):
    '''Output motor voltages smapling from the action space (from env).
    '''
    def __init__(self, env, *args, **kwargs):
        super(RandomControl, self).__init__(env)
        self._action_space = env.action_space

    def action(self, state):
        return self._action_space.sample()


class AeroControl(Control):
    '''Classical controller to set the Quanser Aero back to its original
    position.
    '''
    def __init__(self, env, *args, **kwargs):
        super(AeroControl, self).__init__(env)
        self._desired = np.array([0, 0, 0, 0])
        self._error = np.array([0, 0, 0, 0])
        self._state_x = np.array([0, 0, 0, 0])
        self._gain_p = np.array([98.2088, -103.0645, 32.2643, -29.075])
        self._gain_y = np.array([156.3469, 66.1643, 45.5122, 17.1068])
        self._v_motors = np.array([0, 0])
        self._pitch_n_k1 = 0
        self._pitch_dot_k1 = 0
        self._yaw_n_k1 = 0
        self._yaw_dot_k1 = 0

    def action(self, state):
        # Use only pitch and yaw for control
        pitch_rad = state[0]  # in radians
        yaw_rad = state[1]  # in radians

        self._state_x[0] = pitch_rad
        self._state_x[1] = yaw_rad

        # Z transform 1st order derivative filter start
        pitch_n = pitch_rad
        pitch_dot = (46 * pitch_n) - \
            (46 * self._pitch_n_k1) + (0.839 * self._pitch_dot_k1)
        self._state_x[2] = pitch_dot
        self._pitch_n_k1 = pitch_n
        self._pitch_dot_k1 = pitch_dot
        yaw_n = yaw_rad
        yaw_dot = (46 * yaw_n) - \
            (46 * self._yaw_n_k1) + (0.839 * self._yaw_dot_k1)
        self._state_x[3] = yaw_dot
        self._yaw_n_k1 = yaw_n
        self._yaw_dot_k1 = yaw_dot
        self._error[0] = self._desired[0] - self._state_x[0]
        self._error[1] = self._desired[1] - self._state_x[1]

        # Calculates voltage to be applied to Front and Back Motors K*u
        out_p = 0
        out_y = 0
        for it in range(4):
            out_p = out_p + self._error[it] * self._gain_p[it]
            out_y = out_y + self._error[it] * self._gain_y[it]

        self._v_motors[0] = out_p
        self._v_motors[1] = out_y

        voltages = np.empty(2,)
        voltages[0] = self._v_motors[0]
        voltages[1] = self._v_motors[1]
        # Filter end

        # NOTE: was at 24.0 V for both below
        # Set the saturation limit to +/- AERO_MAX_VOLTAGE for Motor0
        if (voltages[0] > AERO_MAX_VOLTAGE):
            voltages[0] = AERO_MAX_VOLTAGE
        elif (voltages[0] < -AERO_MAX_VOLTAGE):
            voltages[0] = -AERO_MAX_VOLTAGE
        # Set the saturation limit to +/- AERO_MAX_VOLTAGE for Motor1
        if (voltages[1] > AERO_MAX_VOLTAGE):
            voltages[1] = AERO_MAX_VOLTAGE
        elif (voltages[1] < -AERO_MAX_VOLTAGE):
            voltages[1] = -AERO_MAX_VOLTAGE
        # End of Custom Code
        voltages = -voltages

        voltages = np.array(voltages, dtype=np.float64)
        assert voltages.shape == self.action_shape
        return voltages


def unbias(theta):
    return ((theta + np.pi) % (2 * np.pi))


class QubeFlipUpControl(Control):
    '''Classical controller to hold the pendulum upright whenever the
    angle is within 20 degrees, and flips up the pendulum whenever
    outside 20 degrees.
    '''
    def __init__(self, env=None, action_shape=None, sample_freq=1000, **kwargs):
        super(QubeFlipUpControl, self).__init__(env=env)
        self.theta_dot = 0.
        self.alpha_dot = 0.
        self.sample_freq = sample_freq

    def _flip_up(self, theta, alpha, theta_dot, alpha_dot):
        '''Implements a energy based swing-up controller'''
        mu = 50.0 # in m/s/J
        ref_energy = 30.0 / 1000.0 # Er in joules
        max_u = 6.0 # Max action is 6m/s^2

        # System parameters
        jp = 3.3282e-5
        lp = 0.129
        lr = 0.085
        mp = 0.024
        mr = 0.095
        rm = 8.4
        g = 9.81
        kt = 0.042

        pend_torque = (1 / 2) * mp * g * lp * (1 + np.cos(alpha));
        energy = (pend_torque + (jp / 2.0) * alpha_dot * alpha_dot);

        # u = sat_u_max(mu * (E - Er) * sign(alpha_dot * cos(alpha)))
        u = mu * (energy - ref_energy) * np.sign(-1 * np.cos(alpha) * alpha_dot)
        u = np.clip(u, -max_u, max_u)

        torque = (mr * lr) * u
        voltage = (rm / kt) * torque
        return -voltage

    def _action_hold(self, theta, alpha, theta_dot, alpha_dot):
        # multiply by proportional and derivative gains
        kp_theta = -2.0
        kp_alpha = 35.0
        kd_theta = -1.5
        kd_alpha = 3.0
        action = \
            theta * kp_theta + \
            alpha * kp_alpha + \
            theta_dot * kd_theta + \
            alpha_dot * kd_alpha
        return action

    def action(self, state):
        # Get the angles
        theta_x = state[0]
        theta_y = state[1]
        alpha_x = state[2]
        alpha_y = state[3]
        theta = np.arctan2(theta_y, theta_x)
        alpha = np.arctan2(alpha_y, alpha_x)

        # Calculate the velocities
        theta_dot = -2500 * self.theta_dot + 50 * theta
        alpha_dot = -2500 * self.alpha_dot + 50 * alpha

        # If pendulum is within 20 degrees of upright, enable balance control
        if np.abs(alpha) <= (20.0 * np.pi / 180.0):
            action = self._action_hold(theta, alpha, theta_dot, alpha_dot)
        else:
            action = self._flip_up(theta, alpha, theta_dot, alpha_dot)

        self.theta_dot += (-50 * self.theta_dot + theta) / self.sample_freq
        self.alpha_dot += (-50 * self.alpha_dot + alpha) / self.sample_freq

        voltages = np.array([action], dtype=np.float64)
        # set the saturation limit to +/- the Qube saturation voltage
        np.clip(voltages, -QUBE_MAX_VOLTAGE, QUBE_MAX_VOLTAGE, out=voltages)
        assert voltages.shape == self.action_shape
        return voltages


class QubeHoldControl(QubeFlipUpControl):
    '''Classical controller to hold the pendulum upright whenever the
    angle is within 20 degrees. (Same as QubeFlipUpControl but without a
    flip up action)
    '''
    def __init__(self, env, sample_freq=1000, **kwargs):
        super(QubeHoldControl, self).__init__(
            env, sample_freq=sample_freq)

    def _flip_up(self, theta, alpha, theta_dot, alpha_dot):
        return 0
