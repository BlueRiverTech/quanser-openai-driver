# Notes:
# pretty fast! ('Average time of reading state is:   ', 1.05e-06)
#              ('Average time of writing voltage is: ', 1.40e-06)
from __future__ import print_function
from __future__ import division

import numpy as np
import time
from gym_brt.envs.QuanserWrapper import QuanserAero


def time_func(f, *args, **kwargs):
    t = time.time()
    f(*args, **kwargs)
    return time.time() - t


def print_state(currents, encoders, others):
    print("New sample read:")
    print("\t Motor curr: ({}, {})".format(currents[0], currents[1]))
    print("\t Motor enco: ({}, {})".format(encoders[0], encoders[1]))
    print("\t Pitch, Yaw: ({}, {})".format(encoders[2], encoders[3]))
    print("\t Gyro:       ({}, {}, {})".format(
        others[0], others[1], others[2]))
    print("\t Accel:      ({}, {}, {})".format(
        others[3],  others[4], others[5]))
    print("\t Tachometer: ({}, {}, {}, {})".format(
        others[6], others[7], others[8], others[9]))


def test_aero_control():

    # variables for control
    desired = [0, 0, 0, 0]  # 0 => pitch,  1 ==> yaw
    error = [0, 0, 0, 0]
    state_x = [0, 0, 0, 0]
    gain_p = [98.2088, -103.0645, 32.2643, -29.075]
    gain_y = [156.3469, 66.1643, 45.5122, 17.1068]
    v_motors = [0, 0]
    pitch_n_k1 = 0
    pitch_dot_k1 = 0
    yaw_n_k1 = 0
    yaw_dot_k1 = 0

    with QuanserAero(1000) as a:
        voltages = np.array([3.0, 3.0], dtype=np.float64)
        currents, encoders, others = a.action(voltages)

        while True:
            # Start of Custom Code  for LQR 2DOF controller
            encoder_2_deg = 1. * encoders[2] * (360.0 / 2048.0)  # pitch
            # yaw (encoder3 is higher resolution than the other encoders)
            encoder_3_deg = 1. * encoders[3] * (360.0 / 4096.0)
            print_state(currents, encoders, others)

            pitch_rad = 0.01743 * encoder_2_deg
            yaw_rad = 0.01743 * encoder_3_deg
            state_x[0] = pitch_rad
            state_x[1] = yaw_rad

            # Z transform 1st order derivative filter start
            pitch_n = pitch_rad
            pitch_dot = (46*pitch_n) - (46*pitch_n_k1) + (0.839*pitch_dot_k1)
            state_x[2] = pitch_dot
            pitch_n_k1 = pitch_n
            pitch_dot_k1 = pitch_dot
            yaw_n = yaw_rad
            yaw_dot = (46*yaw_n) - (46*yaw_n_k1) + (0.839*yaw_dot_k1)
            state_x[3] = yaw_dot
            yaw_n_k1 = yaw_n
            yaw_dot_k1 = yaw_dot
            error[0] = desired[0] - state_x[0]
            error[1] = desired[1] - state_x[1]

            # Calculates voltage to be applied to Front and Back Motors K*u
            out_p = 0
            out_y = 0
            for it in range(4):
                out_p = out_p + error[it] * gain_p[it]
                out_y = out_y + error[it] * gain_y[it]

            v_motors[0] = out_p
            v_motors[1] = out_y

            voltages[0] = v_motors[0]
            voltages[1] = v_motors[1]

            print("\tvoltages: ({}, {})\n".format(voltages[0], voltages[1]))
            # Filter end

            # NOTE: was at 24.0 V for both below
            # Set the saturation limit to +/- 15v for Motor0
            if (voltages[0] > 15.0):
                voltages[0] = 15.0
            elif (voltages[0] < -15.0):
                voltages[0] = -15.0
            # Set the saturation limit to +/- 15V for Motor1
            if (voltages[1] > 15.0):
                voltages[1] = 15.0
            elif (voltages[1] < -15.0):
                voltages[1] = -15.0
            # End of Custom Code

            voltages = -voltages
            currents, encoders, others = a.action(voltages)


if __name__ == '__main__':
    test_aero_control()
