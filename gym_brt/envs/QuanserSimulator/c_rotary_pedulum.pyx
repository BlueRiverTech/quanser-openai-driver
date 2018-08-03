from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from libc cimport math


# Warning: Implementation is likely buggy and leads to numerical instability
cdef class CythonRotaryPendulumNonLinearApproximation:
    """A Cython implementation of a non-linear pendulum approximation from the
    paper: "On the Dynamics of the Furuta Pendulum" by Benjamin Seth Cazzolato
    and Zebb Prime.

    The constants are found in Quanser's courseware Matlab code.
    To find the original download "Simulink Courseware" from: https://www.quanser.com/products/qube-servo-2/
    The file is under: <download_directory>/Courseware Resources/Fundamentals/Rotary Pendulum/State Space Modeling/Software/setup_ss_model.m
    """
    cdef double rm, kt, km, m1, L1, l1, j1, j1_hat, b1, m2, L2, l2, j2, j2_hat, b2, j0_hat, g

    def __init__(self, **kwargs):
        # Motor
        self.rm = 8.4  # Resistance
        self.kt = 0.042  # Current-torque (N-m/A)
        self.km = 0.042  # Back-emf constant (V-s/rad)

        # Rotary Arm
        self.m1 = 0.095  # Mass of arm (kg)
        self.L1 = 0.085  # Total length of arm (m)
        self.l1 = self.L1 / 2  # Centroid of arm (m)
        self.j1 = self.m1*self.L1**2/12  # Moment of inertia of arm about pivot (kg-m^2)
        self.j1_hat  = self.j1 + self.m1 * (self.l1 ** 2)
        self.b1 = 0.0015  # Equivalent Viscous Damping Coefficient (N-m-s/rad)

        # Pendulum Link
        self.m2 = 0.024  # Mass of pendulum (kg)
        self.L2 = 0.129  # Total length of pendulum (m)
        self.l2 = self.L2 / 2  # Centroid of pendulum (m)
        self.j2 = self.m2*self.L2**2/12  # Moment of inertia of pendulum about pivot (kg-m^2)
        self.j2_hat  = self.j2 + self.m2 * (self.l2 ** 2)
        self.b2 = 0.0005  # Equivalent Viscous Damping Coefficient (N-m-s/rad)
        self.j0_hat = self.j1_hat + self.m2 * self.L1**2

        # Enviroment params
        self.g = 9.81  # Gravity Constant

    def __call__(self, state, input_voltage, time_step, euler_steps=1):
        return self._step(state, input_voltage, time_step, euler_steps=euler_steps)

    cdef _step(self, state, double input_voltage, double time_step, int euler_steps):
        cdef double theta, alpha, theta_dot, alpha_dot, theta_dot_dot, alpha_dot_dot
        cdef double cosalpha, sinalpha, sin2alpha
        cdef double x0, x1, x2, x3, x4, u0, u1, u2, scale

        theta, alpha, theta_dot, alpha_dot = state

        # Use euler's method to approximate the trajectory.
        # Optionally use more steps to make the simulation more accurate.
        time_step /= euler_steps
        for step in range(euler_steps):
            cosalpha = math.cos(alpha)
            sinalpha = math.sin(alpha)
            sin2alpha = math.sin(2 * alpha)

            # Equations 33 and 34 in 'On the Dynamics of the Furuta Pendulum' by Cazzolato and Prime
            scale = 1 / (self.j0_hat * self.j2_hat + self.j2_hat**2 * sinalpha**2 - self.m2**2 * self.L1**2 * self.l2**2 * cosalpha**2)

            x0 = theta_dot
            x1 = alpha_dot
            x2 = theta_dot * alpha_dot
            x3 = theta_dot**2
            x4 = alpha_dot**2

            u0 = (self.km / self.rm) * (input_voltage - self.km * theta_dot) - self.b1 * theta_dot
            u1 = -self.b2 * alpha_dot
            u2 = self.g

            theta_dot_dot = scale * (
                x0  *  -self.j2_hat * self.b1 \
                + x1  *  self.m2 * self.L1 * self.l2 * self.b2 * cosalpha \
                + x2  *  -self.j2_hat**2 * sin2alpha \
                + x3  *  (-1/2) * self.j2_hat * self.m2 * self.L1 * self.l2 * cosalpha * sin2alpha \
                + x4  *  self.j2_hat * self.m2 * self.L1 * self.l2 * sinalpha \
                + u0  *  self.j2_hat \
                + u1  *  -self.m2 * self.L1 * self.l2 * cosalpha \
                + u2  *  (1/2) * self.m2**2 * self.l2**2 * self.L1 * sin2alpha)

            alpha_dot_dot = scale * (
                x0  *  -self.m2 * self.L1 * self.l2 * self.b1 * cosalpha \
                + x1  *  -self.b2 * (self.j0_hat + self.j2_hat * sinalpha**2) \
                + x2  *  self.m2 * self.L1 * self.l2 * self.j2_hat * cosalpha * sin2alpha \
                + x3  *  (-1/2) * sin2alpha * (self.j0_hat * self.j2_hat + self.j2_hat**2 * sinalpha**2) \
                + x4  *  (-1/2) * self.m2**2 * self.L1**2 * sin2alpha \
                + u0  *  -self.m2 * self.L1 * self.l2 * cosalpha \
                + u1  *  self.j0_hat + self.j2_hat * sinalpha**2 \
                + u2  *  -self.m2 * self.l2 * sinalpha * (self.j0_hat + self.j2_hat * sinalpha**2))

            theta = (theta + theta_dot * time_step) % (2 * math.pi)  # theta after update
            alpha = (alpha + alpha_dot * time_step) % (2 * math.pi)  # alpha after update
            theta_dot += theta_dot_dot * time_step  # theta_dot after update
            alpha_dot += alpha_dot_dot * time_step  # alpha_dot after update

        return [theta, alpha, theta_dot, alpha_dot]
