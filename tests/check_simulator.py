from gym_brt.envs import QubeBeginDownEnv, QubeBeginUprightEnv
import numpy as np
import matplotlib.pyplot as plt

# No input
def action_1(state, step_count):
    return np.array([0.0])


# Constant input
def action_2(state, step_count):
    return np.array([-3.0])


# Flip policy
def action_3(state, step_count):
    alpha = np.arctan2(state[3], state[2])
    theta = np.arctan2(state[1], state[0])
    theta_dot = state[4]
    alpha_dot = state[5] + 1e-15
    """Implements a energy based swing-up controller"""
    mu = 50.0  # in m/s/J
    ref_energy = 30.0 / 1000.0  # Er in joules
    max_u = 0.85  # Max action is 6m/s^2

    # System parameters
    jp = 3.3282e-5
    lp = 0.129
    lr = 0.085
    mp = 0.024
    mr = 0.095
    rm = 8.4
    g = 9.81
    kt = 0.042

    pend_torque = (1 / 2) * mp * g * lp * (1 + np.cos(alpha))
    energy = pend_torque + (jp / 2.0) * alpha_dot * alpha_dot

    u = mu * (energy - ref_energy) * np.sign(-1 * np.cos(alpha) * alpha_dot)
    u = np.clip(u, -max_u, max_u)

    torque = (mr * lr) * u
    voltage = (rm / kt) * torque
    return np.array([-voltage])


# Step wave
def action_4(state, step_count):
    if step_count % 85 == 0:
        action = 3.0
    else:
        action = -3.0
    return np.array([action])


# Hold policy
def action_5(state, step):
    alpha = np.arctan2(state[3], state[2])
    theta = np.arctan2(state[1], state[0])
    theta_dot = state[4]
    alpha_dot = state[5]
    # multiply by proportional and derivative gains
    kp_theta = -2.0
    kp_alpha = 35.0
    kd_theta = -1.5
    kd_alpha = 3.0
    action = (
        theta * kp_theta
        + alpha * kp_alpha
        + theta_dot * kd_theta
        + alpha_dot * kd_alpha
    )
    return np.array([action])


# Flip and Hold
def action_6(state, step):
    QUBE_MAX_VOLTAGE = 3.0
    # Get the angles
    theta_x = state[0]
    theta_y = state[1]
    alpha_x = state[2]
    alpha_y = state[3]
    theta = np.arctan2(theta_y, theta_x)
    alpha = np.arctan2(alpha_y, alpha_x)
    theta_dot = state[4]
    alpha_dot = state[5]

    # If pendulum is within 20 degrees of upright, enable balance control
    if np.abs(alpha) <= (20.0 * np.pi / 180.0):
        action = action_5(state, step)
    else:
        action = action_3(state, step)
    return action


def unnormalize_angle(alpha):
    return (alpha + 2 * np.pi) % (2 * np.pi)


def main():

    # Quick check to see if behavior on hardware matches the simulator

    f, (ax1, ax2) = plt.subplots(1, 2)
    alpha_list = []

    # Run policy on simulator
    with QubeBeginDownEnv(use_simulator=True, frequency=500) as env:
        state = env.reset()
        step_count = 0
        for i in range(5000):
            env.render()
            step_count += 1
            action = action_6(state, step_count)
            alpha_list.append(unnormalize_angle(np.arctan2(state[3], state[2])))
            state, _, _, _ = env.step(action)
    ax2.plot(alpha_list)

    # Run policy on hardware
    alpha_list = []
    with QubeBeginDownEnv(use_simulator=False, frequency=500) as env:
        state = env.reset()
        step_count = 0
        for i in range(5000):
            # env.render()
            step_count += 1
            action = action_6(action_index)(state, step_count)
            alpha_list.append(unnormalize_angle(np.arctan2(state[3], state[2])))
            state, _, _, _ = env.step(action)

    ax2.plot(alpha_list)
    plt.show()


if __name__ == "__main__":
    main()
