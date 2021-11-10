try:
    # If you use a device such as a headless raspberry pi this may not work
    from gym.envs.classic_control import rendering
except:
    print("Warning: Can not import classic_control rendering in qube_render.py")

import vpython as vp
import numpy as np


class QubeRendererVypthon:
    def __init__(self, theta, alpha, frequency):
        self.frequency = frequency
        vp.scene.width, vp.scene.height = 1000, 600
        vp.scene.range = 0.25
        vp.scene.title = "QubeServo2-USB rotary pendulum"

        # Dimensions of the rotary pendulum parts
        base_w, base_h, base_d = 0.102, 0.101, 0.102  # width, height, & len of base
        rotor_d, rotor_h = 0.0202, 0.01  # height, diameter of the rotor platform
        rotary_top_l, rotary_top_d = 0.032, 0.012  # height, diameter of the rotary top
        self.arm_l, arm_d = 0.085, 0.00325  # height, diameter of the arm
        self.pen_l, self.pen_d = 0.129, 0.00475  # height, diameter of the pendulum

        # Origin of parts
        arm_origin = vp.vec(0, 0, 0)
        self.rotary_top_origin = vp.vec(0, 0, -rotary_top_l / 2)
        rotor_origin = arm_origin - vp.vec(0, rotor_h + rotary_top_d / 2 - 0.0035, 0)
        base_origin = rotor_origin - vp.vec(0, base_h / 2, 0)

        # Create the part objects
        base = vp.box(
            pos=base_origin,
            size=vp.vec(base_w, base_h, base_d),
            color=vp.vec(0.45, 0.45, 0.45),
        )
        rotor = vp.cylinder(
            pos=rotor_origin,
            axis=vp.vec(0, 1, 0),
            size=vp.vec(rotor_h, rotor_d, rotor_d),
            color=vp.color.yellow,
        )
        self._rotary_top = vp.cylinder(
            pos=self.rotary_top_origin,
            axis=vp.vec(0, 0, 1),
            size=vp.vec(rotary_top_l, rotary_top_d, rotary_top_d),
            color=vp.color.red,
        )
        self._arm = vp.cylinder(
            pos=arm_origin,
            axis=vp.vec(0, 0, 1),
            size=vp.vec(self.arm_l, arm_d, arm_d),
            color=vp.vec(0.7, 0.7, 0.7),
        )
        self._pendulum = vp.cylinder(
            pos=self.pendulum_origin(theta),
            axis=vp.vec(0, 1, 0),
            size=vp.vec(self.pen_l, self.pen_d, self.pen_d),
            color=vp.color.red,
        )

        # Rotate parts to their init orientations
        self._rotary_top.rotate(angle=theta, axis=vp.vec(0, 1, 0), origin=arm_origin)
        self._arm.rotate(angle=theta, axis=vp.vec(0, 1, 0), origin=arm_origin)
        self._pendulum.rotate(
            angle=alpha,
            axis=self.pendulum_axis(theta),
            origin=self.pendulum_origin(theta),
        )
        self.theta, self.alpha = theta, alpha

    def pendulum_origin(self, theta):
        x = self.arm_l * np.sin(theta)
        y = 0
        z = self.arm_l * np.cos(theta)
        return vp.vec(x, y, z)

    def pendulum_axis(self, theta):
        x = np.sin(theta)
        y = 0
        z = np.cos(theta)
        return vp.vec(x, y, z)

    def render(self, theta, alpha):
        dtheta = theta - self.theta
        self.theta = theta

        self._arm.rotate(angle=dtheta, axis=vp.vec(0, 1, 0), origin=vp.vec(0, 0, 0))
        self._rotary_top.rotate(
            angle=dtheta, axis=vp.vec(0, 1, 0), origin=vp.vec(0, 0, 0)
        )
        self._pendulum.origin = self.pendulum_origin(theta)
        self._pendulum.pos = self.pendulum_origin(theta)
        self._pendulum.axis = vp.vec(0, 1, 0)
        self._pendulum.size = vp.vec(self.pen_l, self.pen_d, self.pen_d)
        self._pendulum.rotate(
            angle=alpha,
            axis=self.pendulum_axis(theta),
            origin=self.pendulum_origin(theta),
        )
        vp.rate(self.frequency)

    def close(self, *args, **kwargs):
        pass


class QubeRenderer2D:
    def __init__(self, theta, alpha, frequency):
        width, height = (640, 240)
        self._viewer = rendering.Viewer(width, height)
        l, r, t, b = (2, -2, 0, 100)
        theta_poly = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
        l, r, t, b = (2, -2, 0, 100)
        alpha_poly = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
        theta_circle = rendering.make_circle(radius=100, res=64, filled=False)
        theta_circle.set_color(0.5, 0.5, 0.5)  # Theta is grey
        alpha_circle = rendering.make_circle(radius=100, res=64, filled=False)
        alpha_circle.set_color(0.8, 0.0, 0.0)  # Alpha is red
        theta_origin = (width / 2 - 150, height / 2)
        alpha_origin = (width / 2 + 150, height / 2)
        self._theta_tx = rendering.Transform(translation=theta_origin)
        self._alpha_tx = rendering.Transform(translation=alpha_origin)
        theta_poly.add_attr(self._theta_tx)
        alpha_poly.add_attr(self._alpha_tx)
        theta_circle.add_attr(self._theta_tx)
        alpha_circle.add_attr(self._alpha_tx)
        self._viewer.add_geom(theta_poly)
        self._viewer.add_geom(alpha_poly)
        self._viewer.add_geom(theta_circle)
        self._viewer.add_geom(alpha_circle)

    def render(self, theta, alpha):
        self._theta_tx.set_rotation(self._theta + np.pi)
        self._alpha_tx.set_rotation(self._alpha)
        return self._viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        self._viewer.close()
