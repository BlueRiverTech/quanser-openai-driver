from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import time
import math
import numpy as np

from gym import spaces
from gym.utils import seeding

from gym_brt.control import flip_and_hold_policy, dampen_policy

# For other platforms where it's impossible to install the HIL SDK
try:
    from gym_brt.quanser.quanser_wrapper.quanser_wrapper import QubeServo2
except ImportError:
    print("Warning: Can not import QubeServo2 in qube_interface.py")

from gym_brt.quanser.qube_simulator import forward_model_euler, forward_model_ode


class QubeHardware(object):
    """Simplify the interface between the environment and """

    def __init__(self, frequency=250, max_voltage=18.0):
        self._theta_dot_cstate = 0
        self._alpha_dot_cstate = 0
        self._frequency = frequency

        # Open the Qube
        self.qube = QubeServo2(frequency=frequency)  # TODO: max_voltage=max_voltage
        self.qube.__enter__()

        self.state = np.array([0, 0, 0, 0], dtype=np.float64)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _match_state(s):
        theta, alpha, theta_dot, alpha_dot = s
        return np.array([])

    def step(self, action, led=None):
        theta, alpha, theta_dot, alpha_dot = self.state
        currents, encoders, others = self.qube.action(action, led_w=led)

        # Calculate alpha, theta, alpha_dot, and theta_dot
        theta = encoders[0] * (-2.0 * np.pi / 2048)
        # Alpha without normalizing
        alpha_un = encoders[1] * (2.0 * np.pi / 2048)
        # Normalized and shifted alpha
        alpha = (alpha_un % (2.0 * np.pi)) - np.pi

        # fmt: off
        theta_dot = -2500 * self._theta_dot_cstate + 50 * theta
        alpha_dot = -2500 * self._alpha_dot_cstate + 50 * alpha_un
        self._theta_dot_cstate += (-50 * self._theta_dot_cstate + theta) / self._frequency
        self._alpha_dot_cstate += (-50 * self._alpha_dot_cstate + alpha_un) / self._frequency
        # fmt: on

        state = np.array([theta, alpha, theta_dot, alpha_dot], dtype=np.float64)
        return state

    def reset_up(self):
        """Run classic control for flip-up until the pendulum is inverted for
        a set amount of time. Assumes that initial state is stationary
        downwards.
        """
        time_hold = 1.0 * self._frequency  # Hold upright 1 second
        sample = 0  # Samples since control system started
        samples_upright = 0  # Consecutive samples pendulum is upright

        action = np.array([1], dtype=np.float64)
        state = self.step([1.0])
        while True:
            action = flip_and_hold_policy(state)
            state = self.step(action)

            # Break if pendulum is inverted
            alpha = state[1]
            if alpha < (10 * np.pi / 180):
                if samples_upright > time_hold:
                    break
                samples_upright += 1
            else:
                samples_upright = 0
            sample += 1

        return state

    def reset_down(self):
        action = np.array([0], dtype=np.float64)
        time_hold = 0.5 * self._frequency  # 0.5 seconds
        samples_downwards = 0  # Consecutive samples pendulum is stationary

        state = self.step([1.0])
        while True:
            action = dampen_policy(state)
            state = self.step(action)

            # Break if pendulum is stationary
            alpha = state[1]
            if abs(alpha) > (178 * np.pi / 180):
                if samples_downwards > time_hold:
                    break
                samples_downwards += 1
            else:
                samples_downwards = 0
        return state

    def reset_encoders(self):
        """Fully stop the pendulum at the bottom. Then reset the alpha encoder"""
        self.reset_down()

        self.step(np.array([0], dtype=np.float64))
        print("Doing a hard reset and, reseting the alpha encoder")
        time.sleep(25)  # Do nothing for 3 seconds to ensure pendulum is stopped

        # This is needed to prevent sensor drift on the alpha/pendulum angle
        # We ONLY reset the alpha channel because the dampen function stops the
        # pendulum from moving but does not perfectly center the pendulum at the
        # bottom (this way alpha is very close to perfect and theta does not
        # drift much)
        print("Pre encoder reset:", self.qube.action(np.array([0], dtype=np.float64)))
        self.qube.reset_encoders(channels=[1])  # Alpha channel only
        print("After encoder reset:", self.qube.action(np.array([0], dtype=np.float64)))

    def close(self, type=None, value=None, traceback=None):
        # Safely close the Qube
        self.qube.__exit__(type=type, value=value, traceback=traceback)


class QubeSimulator(object):
    """Simulator that has the same interface as the hardware wrapper."""

    def __init__(
        self, forward_model="ode", frequency=250, integration_steps=1, max_voltage=18.0
    ):
        if isinstance(forward_model, str):
            if forward_model == "ode":
                self._forward_model = forward_model_ode
            elif forward_model == "euler":
                self._forward_model = forward_model_euler
            else:
                raise ValueError(
                    "'forward_model' must be one of ['ode', 'euler'] or a callable."
                )
        elif callable(forward_model):
            self._forward_model = forward_model
        else:
            raise ValueError(
                "'forward_model' must be one of ['ode', 'euler'] or a callable."
            )

        self._dt = 1.0 / frequency
        self._integration_steps = integration_steps
        self._max_voltage = max_voltage
        self.state = (
            np.array([0, 0, 0, 0], dtype=np.float64) + np.random.randn(4) * 0.01
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def step(self, action, led=None):
        action = np.clip(action, -self._max_voltage, self._max_voltage)
        self.state = self._forward_model(
            *self.state, action, self._dt, self._integration_steps
        )
        return self.state

    def reset_up(self):
        self.state = (
            np.array([0, 0, 0, 0], dtype=np.float64) + np.random.randn(4) * 0.01
        )
        return self.state

    def reset_down(self):
        self.state = (
            np.array([0, np.pi, 0, 0], dtype=np.float64) + np.random.randn(4) * 0.01
        )
        return self.state

    def reset_encoders(self):
        pass

    def close(self, type=None, value=None, traceback=None):
        pass
