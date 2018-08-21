from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


# Set the motor saturation limits for the Aero and Qube
AERO_MAX_VOLTAGE = 15.0
QUBE_MAX_VOLTAGE = 5.0


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


class QubeFlipUpControl(Control):
    '''Classical controller to hold the pendulum upright whenever the
    angle is within 30 degrees, and flips up the pendulum whenever
    outside 30 degrees.
    '''
    def __init__(self, env=None, action_shape=None, sample_freq=1000, **kwargs):
        super(QubeFlipUpControl, self).__init__(env=env)
        self.theta_dot_filtered = 0.
        self.alpha_dot_filtered = 0.
        self.theta = 0.
        self.alpha = 0.
        self._sample_freq = sample_freq

    def _flip_up(self, theta, alpha, theta_dot, alpha_dot):
        # Found analytically
        K = np.array(
            [3.1622776602, 20.7014465019, 1.9988564038, 2.4570771444])
        state = np.array([alpha, alpha_dot, theta, theta_dot])
        action = np.dot(state, K)
        return action

    def _action_hold(self, theta, alpha):
        self.theta_dot_filtered = self.theta * 50.0 - theta * 50.0 + \
            np.exp(-50.0 * self._sample_freq) * self.theta_dot_filtered

        self.alpha_dot_filtered = self.alpha * 50.0 - alpha * 50.0 + \
            np.exp(-50.0 * self._sample_freq) * self.alpha_dot_filtered

        # multiply by proportional and derivative gains
        kp_theta = 2.0
        kd_theta = -2.0
        kp_alpha = -30.0
        kd_alpha = 2.5
        motor_voltage = self.theta * kp_theta + \
            self.theta_dot_filtered * kd_theta + \
            self.alpha * kp_alpha + \
            self.alpha_dot_filtered * kd_alpha

        # Invert for positive CCW
        motor_voltage = -motor_voltage

        return motor_voltage

    def action(self, state):
        theta_x = state[0]
        theta_y = state[1]
        alpha_x = state[2]
        alpha_y = state[3]
        theta = np.arctan2(theta_y, theta_x)
        alpha = np.arctan2(alpha_y, alpha_x)
        theta_dot = (theta - self.theta) * self._sample_freq
        alpha_dot = (alpha - self.alpha) * self._sample_freq

        # If pendulum is within 20 degrees of upright, enable balance control
        if np.abs(alpha) <= (20.0 * np.pi / 180.0):
            motor_voltage = self._action_hold(theta, alpha)
        else:
            motor_voltage = self._flip_up(theta, alpha, theta_dot, alpha_dot)

        self.alpha = alpha
        self.theta = theta

        voltages = np.array([motor_voltage], dtype=np.float64)

        # set the saturation limit to +/- the Qube saturation voltage
        np.clip(voltages, -QUBE_MAX_VOLTAGE, QUBE_MAX_VOLTAGE, out=voltages)

        assert voltages.shape == self.action_shape
        return voltages


class QubeHoldControl(QubeFlipUpControl):
    '''Classical controller to hold the pendulum upright whenever the
    angle is within 30 degrees. (Same as QubeFlipUpControl but without a
    flip up action)
    '''
    def __init__(self, env, sample_freq=1000, **kwargs):
        super(QubeHoldControl, self).__init__(
            env, sample_freq=sample_freq)

    def _flip_up(self, theta, alpha, theta_dot, alpha_dot):
        return 0
