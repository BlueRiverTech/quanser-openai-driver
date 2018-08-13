from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import time
import math
import numpy as np

from gym import spaces
from gym.utils import seeding
from gym_brt.envs.QuanserWrapper import QubeServo2
from gym_brt.envs.QuanserSimulator.quanser_simulator import QuanserSimulator

from gym_brt.control import QubeFlipUpInvertedClassicControl


# theta, alpha: positions, velocities, accelerations
OBSERVATION_HIGH = np.asarray([
    1, 1, 1, 1,  # angles
    np.pi / 4, np.pi / 4,  # velocities
    np.pi / 4, np.pi / 4,  # accelerations
    4100,  # tach0
    0.2,  # sense
], dtype=np.float64)
OBSERVATION_LOW = -OBSERVATION_HIGH

MAX_MOTOR_VOLTAGE = 8.0
ACTION_HIGH = np.asarray([MAX_MOTOR_VOLTAGE], dtype=np.float64)
ACTION_LOW = -ACTION_HIGH

WARMUP_TIME = 5 # in seconds

STATE_KEYS = [
    'COS_THETA',
    'SIN_THETA',
    'COS_ALPHA',
    'SIN_ALPHA',
    'THETA_VELOCITY',
    'ALPHA_VELOCITY',
    'THETA_ACCELERATION',
    'ALPHA_ACCELERATION',
    'TACH0',
    'SENSE'
]

QUBES = {
    'QubeServo2': QubeServo2,
    'QubeSimLinear': lambda frequency: QuanserSimulator(pendulum='RotaryPendulumLinearApproximation', safe_operating_voltage=18.0, euler_steps=1, frequency=frequency),
    'QubeSimNonLinear': lambda frequency: QuanserSimulator(pendulum='RotaryPendulumNonLinearApproximation', safe_operating_voltage=18.0, euler_steps=1, frequency=frequency),
    'QubeSimNonLinearCython': lambda frequency: QuanserSimulator(pendulum='CythonRotaryPendulumNonLinearApproximation', safe_operating_voltage=18.0, euler_steps=1, frequency=frequency),
}


def normalize_angle(theta):
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


class QubeBaseReward(object):
    def __init__(self):
        self.target_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH, dtype=np.float32)

    def __call__(self, state, action):
        raise NotImplementedError


class QubeBaseEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, env_base='QubeServo2', frequency=1000, alpha_tolerance=(10 * np.pi / 180)):
        self.observation_space = spaces.Box(
            OBSERVATION_LOW, OBSERVATION_HIGH,
            dtype=np.float32)

        self.action_space = spaces.Box(
            ACTION_LOW, ACTION_HIGH,
            dtype=np.float32)

        self._alpha_tolerance = alpha_tolerance

        self.reward_fn = QubeBaseReward()

        self._theta = 0
        self._alpha = 0
        self._theta_velocity = 0
        self._alpha_velocity = 0
        self._theta_acceleration = 0
        self._alpha_acceleration = 0
        self._motor_voltage = 0
        self._tach0 = 0
        self._sense = 0
        self._frequency = frequency
        self._prev_t = time.time()

        # Open the Qube
        self.qube = QUBES[env_base](frequency=frequency)
        self.qube.__enter__()

        self.seed()
        self.viewer = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        motor_voltages = np.clip(np.array([action[0]], dtype=np.float64),
                                 ACTION_LOW, ACTION_HIGH)
        currents, encoders, others = self.qube.action(motor_voltages)

        encoder0 = encoders[0]
        encoder1 = encoders[1] % 2048
        if encoder1 < 0:
            encoder1 = 2048 + encoder1

        theta_next = encoder0 * (-2.0 * np.pi / 2048)
        alpha_next = encoder1 * (2.0 * np.pi / 2048) - np.pi
        self._sense = currents[0]
        self._tach0 = others[0]

        theta_velocity_next = normalize_angle(theta_next - self._theta)
        alpha_velocity_next = normalize_angle(alpha_next - self._alpha)

        self._theta_acceleration = normalize_angle(
            theta_velocity_next - self._theta_velocity)
        self._alpha_acceleration = normalize_angle(
            alpha_velocity_next - self._alpha_velocity)

        self._theta_velocity = theta_velocity_next
        self._alpha_velocity = alpha_velocity_next

        self._theta = theta_next
        self._alpha = alpha_next

        return self._get_state()

    def _get_state(self):
        state = np.asarray([
            np.cos(self._theta),
            np.sin(self._theta),
            np.cos(self._alpha),
            np.sin(self._alpha),
            self._theta_velocity,
            self._alpha_velocity,
            self._theta_acceleration,
            self._alpha_acceleration,
            self._tach0,
            self._sense,
        ], dtype=np.float32)
        return state

    def _flip_up(self, early_quit=False, time_out=5, min_hold_time=1):
        """Run classic control for flip-up until the pendulum is inverted for a
        set amount of time. Assumes that initial state is stationary downwards.

        Args:
            early_quit: Quit if flip up doesn't succeed after set amount of time
            time_out: Time given to the classical control system to flip up 
                before quitting (in seconds)
            min_hold_time: Time to hold the pendulum upright within a tolerance
                (in seconds)
            alpha_tolerance: Angle from perfectly inverted that counts as
                'upright' (in radians)
        """
        control = QubeFlipUpInvertedClassicControl(env=self, sample_freq=self._frequency)
        time_out = time_out * self._frequency
        time_hold = min_hold_time * self._frequency
        sample = 0 # Samples since control system started
        samples_upright = 0 # Consecutive samples pendulum is upright

        state = self._get_state()
        while True:
            action = control.action(state)
            state, _, _, _ = self.step(action)

            # Break or reset to down if timed out
            if sample > time_out:
                if early_quit:
                    break
                else:
                    sample = 0
                    samples_upright = 0
                    self.dampen_down()

            # Break if pendulum is inverted
            if self._alpha < self._alpha_tolerance:
                if samples_upright > time_hold:
                    break
                samples_upright += 1
            else:
                samples_upright = 0
            sample += 1

        return state

    def _dampen_down(self, min_hold_time=0.5):
        action = np.zeros(
            shape=self.action_space.shape,
            dtype=self.action_space.dtype)

        time_hold = min_hold_time * self._frequency
        samples_downwards = 0 # Consecutive samples pendulum is downwards

        while True:
            state, _, _, _ = self.step(action)
            # Break if pendulum is stationary
            ref_state = [0.0, 0.0, 0.0, 0.0]
            if np.allclose(state[4:8], ref_state, rtol=1e-02, atol=1e-03):
                if samples_downwards > time_hold:
                    # self.state_offsets[4:] = state[4:]
                    break
                samples_downwards += 1
            else:
                samples_downwards = 0

        return self._get_state()

    def _center(self):
        return self._get_state()

    def flip_up(self, early_quit=False, time_out=np.inf, min_hold_time=1):
        self.dampen_down()
        return self._flip_up(
            early_quit=early_quit,
            time_out=time_out,
            min_hold_time=min_hold_time)

    def dampen_down(self):
        self.center()
        return self._dampen_down()

    def center(self):
        return self._center()

    def reset(self):
        # Start the pendulum stationary at the bottom (stable point)
        state = self._dampen_down()
        self.qube.reset_encoders()
        action = np.zeros(
            shape=self.action_space.shape,
            dtype=self.action_space.dtype)
        return self.step(action)[0]

    def step(self, action):
        state = self._step(action)
        reward = self.reward_fn(state, action)
        done = False

        info = {k: v for (k, v) in zip(STATE_KEYS, self._get_state())}
        info['alpha'] = self._alpha
        info['theta'] = self._theta

        return state, reward, done, info

    def render(self, mode='human'):
        # Simple and *NOT* physically accurate rendering
        screen = screen_width = screen_height = 600
        scale = 0.5 * screen / 100.0 # Everything is scaled out of 100

        qubewidth = 10.0 * scale
        qubeheight = 10.0 * scale
        origin = (screen_width/2, screen_height/2)

        arm_len = 40 * scale
        arm_width = 1.0 * scale

        pen_len = 40 * scale
        pen_width = 2.0 * scale

        def pen_origin(theta, origin=origin, len=arm_len,):
            x = origin[0] - len * math.sin(theta)
            y = origin[1] + len * math.cos(theta)
            return x, y

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # draw qube base
            l,r,t,b = qubewidth/2, -qubewidth/2, -qubeheight/2, qubeheight/2
            qube = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            qube.set_color(0., 0., 0.)
            qubetrans = rendering.Transform(translation=origin)
            qube.add_attr(qubetrans)
            self.viewer.add_geom(qube)

            # draw qube arm
            l,r,t,b = arm_width/2, -arm_width/2, 0, arm_len
            arm = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            arm.set_color(.5,.5,.5)
            self.armtrans = rendering.Transform(translation=origin)
            arm.add_attr(self.armtrans)
            self.viewer.add_geom(arm)

            arm_trace = rendering.make_circle(radius=arm_len, filled=False)
            armtracetrans = rendering.Transform(translation=origin)
            arm_trace.set_color(.5,.5,.5) 
            arm_trace.add_attr(armtracetrans)
            self.viewer.add_geom(arm_trace)

            # draw qube pendulum
            pen_orgin = (origin[0], origin[1] + arm_len)
            l,r,t,b = pen_width/2, -pen_width/2, 0, pen_len
            pen = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pen.set_color(1., 0., 0.)
            self.pentrans = rendering.Transform(translation=pen_orgin, rotation=math.pi/10)
            pen.add_attr(self.pentrans)
            self.viewer.add_geom(pen)

        self.armtrans.set_rotation(np.pi+self._theta)
        self.pentrans.set_translation(*pen_origin(np.pi+self._theta))
        self.pentrans.set_rotation(self._alpha)

        self.viewer.render(return_rgb_array = 'human'=='rgb_array')
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        # Safely close the Qube
        self.qube.__exit__(None, None, None)
        if self.viewer: self.viewer.close()
