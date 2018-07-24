from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


class QuanserQubeSimulator:
    def __init__(self, frequency=1000):
        """
        Args:
            - Frequency: Frequency of the reading/writing task (in Hz)
        """
        self._frequency = frequency

        self._theta = 0
        self._alpha = np

        length_arm = 0.00  # Lr
        length_pendulum = 0.00  # Lp

        terminal_resistance = 8.4  # Rm Terminal resistance 8.4Ω
        torque_const = 0.042  # kt = Torque constant 0.042 N.m/A
        motor_backemf = 0.042  # km Motor back-emf constant 0.042 V/(rad/s)
        rotor_inertia = 4.0e-6  # Jm Rotor inertia 4.0 × 10 −6 kg.m 2
        rotor_inductance = 1.16  # Lm Rotor inductance 1.16 mH
        0.0106  # mh Load hub mass 0.0106 kg
        0.0111  # rh Load hub mass 0.0111 m
        0.6e-6  # Jh Load hub inertia 0.6 × 10 −6 kg.m 2
        0.053  # md Mass of disk load 0.053 kg
        0.0248  # rd Radius of disk load 0.0248 m

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def action(self, voltages_w):
        return currents_r, encoder_r_buffer, other_r_buffer


    def _step(state, action, time_interval=None):
        pass


    # Equations
    def moment_of_inertia_motor_shaft(motor_backemf, current_motor)
        # tau_m = km * im(t)
        return motor_backemf * current

    def current_motor(voltage_motor_input, motor_backemf, motor_speed, terminal_resistance):
        # im(t) = (vm(t) − km * omega_m(t)) / Rm
        return (voltage_motor_input - motor_backemf * motor_speed) / terminal_resistance
        