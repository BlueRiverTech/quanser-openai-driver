from __future__ import print_function
from __future__ import division

import numpy as np
import time
from gym_brt.envs.QuanserWrapper import QubeServo2


def time_func(f, *args, **kwargs):
    t = time.time()
    f(*args, **kwargs)
    return time.time() - t


def print_state(currents, encoders, others):
    current0 = currents[0]
    motor0_position = encoders[0]
    encoder1_position = encoders[1]
    tach0 = others[0]

    print("\nNew Read")
    print("\tmotor0 current:", current0)
    print("\tmotor0 enco:", motor0_position)
    print("\tmotor1 enco:", encoder1_position)
    print("\ttach0:", tach0)


def test_py_control():
    M_PI = 3.14159

    # global variables for the pendulum balance control
    theta_n_k1 = 0
    theta_dot_k1 = 0
    alpha_n_k1 = 0
    alpha_dot_k1 = 0

    with QubeServo2(25) as a:
        voltages = np.array([0.], dtype=np.float64)
        currents, encoders, others = a.action(voltages)

        while True:
            # Start Pendulum Code
            encoder0 = encoders[0]
            encoder1 = encoders[1] % 2048
            if (encoder1 < 0):
                encoder1 = 2048 + encoder1
            theta = encoder0 * (-2.0 * M_PI / 2048)
            alpha = encoder1 * (2.0 * M_PI / 2048) - M_PI
            current_sense = currents[0]

            print_state(currents, encoders, others)

            # Start of Custom Code for controller
            # if the pendulum is within +/-30 degrees of upright, enable balance
            # control
            if np.abs(alpha) <= (30.0 * M_PI / 180.0):
                # transfer function = 50s/(s+50)
                # z-transform at 1ms = (50z - 50)/(z-0.9512)
                theta_n = -theta
                theta_dot = (50.0 * theta_n) - \
                    (50.0 * theta_n_k1) + (0.9512 * theta_dot_k1)
                theta_n_k1 = theta_n
                theta_dot_k1 = theta_dot

                # transfer function = 50s/(s+50)
                # z-transform at 1ms = (50z - 50)/(z-0.9512)
                alpha_n = -alpha
                alpha_dot = (50.0 * alpha_n) - \
                    (50.0 * alpha_n_k1) + (0.9512 * alpha_dot_k1)
                alpha_n_k1 = alpha_n
                alpha_dot_k1 = alpha_dot

                # multiply by proportional and derivative gains
                kp_theta = 2.0
                kd_theta = -2.0
                kp_alpha = -30.0
                kd_alpha = 2.5
                motor_voltage = (theta * kp_theta) + (theta_dot * kd_theta) + \
                    (alpha * kp_alpha) + (alpha_dot * kd_alpha)

                # set the saturation limit to +/- 15V
                if motor_voltage > 15.0:
                    motor_voltage = 15.0
                elif motor_voltage < -15.0:
                    motor_voltage = -15.0

                # invert for positive CCW
                motor_voltage = -motor_voltage
            else:
                motor_voltage = 0
            # End of Pendulum Code
            voltages = np.array([motor_voltage], dtype=np.float64)
            currents, encoders, others = a.action(voltages)


if __name__ == '__main__':
    test_py_control()
