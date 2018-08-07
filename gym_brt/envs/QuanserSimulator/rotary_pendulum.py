from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import math


# Warning: Implementation is likely buggy and leads to numerical instability
class RotaryPendulumNonLinearApproximation(object):
    """A non-linear pendulum approximation from the paper: "On the Dynamics of
    the Furuta Pendulum" by Benjamin Seth Cazzolato and Zebb Prime.

    The constants are found in Quanser's courseware Matlab code.
    To find the original download "Simulink Courseware" from: https://www.quanser.com/products/qube-servo-2/
    The file is under: <download_directory>/Courseware Resources/Fundamentals/Rotary Pendulum/State Space Modeling/Software/setup_ss_model.m
    """
    def __init__(self):
        # Motor
        self.rm = 8.4  # Resistance
        self.kt = 0.042  # Current-torque (N-m/A)
        self.km = 0.042  # Back-emf constant (V-s/rad)

        # Rotary Arm
        self.m1 = 0.095  # Mass of arm (kg)
        self.L1 = 0.085  # Total length of arm (m)
        self.l1 = self.L1 / 2  # Centroid of arm (m)
        self.j1 = self.m1*self.L1**2/12  # Moment of inertia of arm about pivot (kg-m^2)
        self.j1_hat  = self.j1 + self.m1 * self.l1**2
        self.b1 = 0.0015  # Equivalent Viscous Damping Coefficient (N-m-s/rad)

        # Pendulum Link
        self.m2 = 0.024  # Mass of pendulum (kg)
        self.L2 = 0.129  # Total length of pendulum (m)
        self.l2 = self.L2 / 2  # Centroid of pendulum (m)
        self.j2 = self.m2*self.L2**2/12  # Moment of inertia of pendulum about pivot (kg-m^2)
        self.j2_hat  = self.j2 + self.m2 * self.l2**2
        self.b2 = 0.0005  # Equivalent Viscous Damping Coefficient (N-m-s/rad)
        self.j0_hat = self.j1_hat + self.m2 * self.L1**2

        # Enviroment params
        self.g = 9.81  # Gravity Constant

    def __call__(self, state, action, time_step, euler_steps=1):
        theta, alpha, theta_dot, alpha_dot = state

        time_step = time_step / euler_steps
        for step in range(euler_steps):
            cosalpha = math.cos(alpha)
            sinalpha = math.sin(alpha)
            sin2alpha = math.sin(2 * alpha)

            # Equations 33 and 34 in 'On the Dynamics of the Furuta Pendulum' by Cazzolato and Prime
            scale = 1 / (self.j0_hat * self.j2_hat + self.j2_hat**2 * sinalpha**2 - self.m2**2 * self.L1**2 * self.l2**2 * cosalpha**2)
            A1 = np.array([
                    -self.j2_hat * self.b1,
                    self.m2 * self.L1 * self.l2 * self.b2 * cosalpha,
                    -self.j2_hat**2 * sin2alpha,
                    (-1/2) * self.j2_hat * self.m2 * self.L1 * self.l2 * cosalpha * sin2alpha,
                    self.j2_hat * self.m2 * self.L1 * self.l2 * sinalpha
                ]).reshape(1,5)
            B1 = np.array([
                    self.j2_hat,
                    -self.m2 * self.L1 * self.l2 * cosalpha,
                    (1/2) * self.m2**2 * self.l2**2 * self.L1 * sin2alpha
                ]).reshape(1,3)
            A2 = np.array([
                    self.m2 * self.L1 * self.l2 * self.b1 * cosalpha,
                    -self.b2 * (self.j0_hat + self.j2_hat * sinalpha**2),
                    self.m2 * self.L1 * self.l2 * self.j2_hat * cosalpha * sin2alpha,
                    (-1/2) * sin2alpha * (self.j0_hat * self.j2_hat + self.j2_hat**2 * sinalpha**2),
                    (-1/2) * self.m2**2 * self.L1**2 * sin2alpha
                ]).reshape(1,5)
            B2 = np.array([
                    -self.m2 * self.L1 * self.l2 * cosalpha,
                    self.j0_hat + self.j2_hat * sinalpha**2,
                    -self.m2 * self.l2 * sinalpha * (self.j0_hat + self.j2_hat * sinalpha**2)
                ]).reshape(1,3)
            x = np.array([
                    [theta_dot],
                    [alpha_dot],
                    [theta_dot * alpha_dot],
                    [theta_dot**2], 
                    [alpha_dot**2]
                ]).reshape(5,1)
            u = np.array([
                    [(self.km / self.rm) * (action - self.km * theta_dot) - self.b1 * theta_dot],
                    [-self.b2 * alpha_dot],
                    [self.g]
                ]).reshape(3,1)

            theta_dot_dot = np.asscalar(scale * (A1.dot(x) + B1.dot(u)))
            alpha_dot_dot = np.asscalar(scale * (A2.dot(x) + B2.dot(u)))

            theta = (theta + theta_dot * time_step) % (2 * math.pi)  # theta after update
            alpha = (alpha + alpha_dot * time_step) % (2 * math.pi)  # alpha after update
            theta_dot += theta_dot_dot * time_step  # theta_dot after update
            alpha_dot += alpha_dot_dot * time_step  # alpha_dot after update

        state = [theta, alpha, theta_dot, alpha_dot]
        return state

class RotaryPendulumLinearApproximation(object):
    """A linear pendulum approximation from QUBE Servo 2 Workbook: State Space Modeling

    The constants are found in Quanser's courseware Matlab code.
    To find the original download "Simulink Courseware" from: https://www.quanser.com/products/qube-servo-2/
    The file is under: <download_directory>/Courseware Resources/Fundamentals/Rotary Pendulum/State Space Modeling/Software/setup_ss_model.m
    """
    def __init__(self):
        # Motor
        Rm = 8.4  # Resistance
        kt = 0.042  # Current-torque (N-m/A)
        km = 0.042  # Back-emf constant (V-s/rad)

        # Rotary Arm
        Mr = 0.095  # Mass (kg)
        Lr = 0.095  # Total length (m)
        Jr = Mr * (Lr**2) / 12  # Moment of inertia about pivot (kg-m^2)
        Dr = 0.0015  # Equivalent Viscous Damping Coefficient (N-m-s/rad)

        # Pendulum Link
        Mp = 0.024  # Mass (kg)
        Lp = 0.129  # Total length (m)
        Jp = Mp * (Lp**2) / 12  # Moment of inertia about pivot (kg-m^2)
        Dp = 0.0005  # Equivalent Viscous Damping Coefficient (N-m-s/rad)
        g = 9.81  # Gravity Constant

        # State Space Representation
        Jt = Jr*Jp + Mp*(Lp/2)**2*Jr + Jp*Mp*Lr**2
        self.A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, Mp**2*(Lp/2)**2*Lr*g/Jt, -Dr*(Jp+Mp*(Lp/2)**2)/Jt, Mp*(Lp/2)*Lr*Dp/Jt],
            [0, -Mp*g*(Lp/2)*(Jr+Mp*Lr**2)/Jt, Mp*(Lp/2)*Lr*Dr/Jt, -Dp*(Jr+Mp*Lr**2)/Jt],
        ])
        self.B = np.array([0, 0, (Jp+Mp*(Lp/2)**2)/Jt, Mp*(Lp/2)*Lr/Jt]).reshape((4,1))

        # Add actuator dynamics
        self.B = kt * self.B / Rm
        self.A[2,2] = self.A[2,2] - kt*kt/Rm*self.B[2]
        self.A[3,2] = self.A[3,2] - kt*kt/Rm*self.B[3]
        
    def __call__(self, state, action, time_step, euler_steps=None):
        # Euler steps are ignored
        state = state.reshape(4, 1)
        state_dot = self.A.dot(state) + self.B * action
        state_delta = state_dot * time_step
        state += state_delta
        return state.reshape(4,)

