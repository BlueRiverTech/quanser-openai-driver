try:
    import autograd.numpy as np
    from autograd import grad, jacobian
except:
    raise ImportError("Please install autograd.")
try:
    import scipy.linalg as sp_linalg
except:
    raise ImportError("Please install scipy.")


# Motor
Rm = 8.4  # Resistance
kt = 0.042  # Current-torque (N-m/A)
km = 0.042  # Back-emf constant (V-s/rad)

# Rotary Arm
mr = 0.095  # Mass (kg)
Lr = 0.085  # Total length (m)
Jr = mr * Lr ** 2 / 12  # Moment of inertia about pivot (kg-m^2)
Br = Dr = 0.0015  # Equivalent viscous damping coefficient (N-m-s/rad)

# Pendulum Link
mp = 0.024  # Mass (kg)
Lp = 0.129  # Total length (m)
Jp = mp * Lp ** 2 / 12  # Moment of inertia about pivot (kg-m^2)
Dp = Bp = 0.0005  # Equivalent viscous damping coefficient (N-m-s/rad)

g = 9.81  # Gravity constant


def forward_model(state, action, dt=1 / 300.0):
    theta = state[0]
    alpha = state[1]
    theta_dot = state[2]
    alpha_dot = state[3]
    Vm = action

    tau = (km * (Vm - km * theta_dot)) / Rm  # torque

    # fmt: off
    alpha_dot_dot = (2.0*Lp*Lr*mp*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*np.sin(2.0*alpha) + 2.0*Lp*Lr*alpha_dot**2*mp*np.sin(alpha) - 4.0*(tau))*np.cos(alpha) - 0.5*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp)*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha)))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp))
    theta_dot_dot = (-Lp*Lr*mp*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha))*np.cos(alpha) + (4.0*Jp + Lp**2*mp)*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*np.sin(2.0*alpha) + 2.0*Lp*Lr*alpha_dot**2*mp*np.sin(alpha) - 4.0*(tau)))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp))
    # fmt: on

    # Works around a single operating point
    theta += theta_dot * dt  # TODO: Shouldn't the angles be ahead of the velocities?
    alpha += alpha_dot * dt
    theta_dot += theta_dot_dot * dt
    alpha_dot += alpha_dot_dot * dt

    theta %= 2 * np.pi
    alpha %= 2 * np.pi

    # For continuous version of LQR
    # state = np.array([theta_dot,alpha_dot,theta_dot_dot,alpha_dot_dot]).reshape((4,))

    # For discrete version of LQR
    state = np.array([theta, alpha, theta_dot, alpha_dot]).reshape((4,))
    return state


def computeAB(current_state, current_control):
    # Linearizing Dynamics
    forward_dynamics_model = lambda state, action: forward_model(state, action)
    a_mat = jacobian(forward_dynamics_model, 0)
    b_mat = jacobian(forward_dynamics_model, 1)
    A = a_mat(current_state, current_control)
    B = b_mat(current_state, current_control)

    # Correct continuous time linearization from Quanser Workbook -
    # A = np.array( [[0,0,1.0000,0],[0,0,0 ,1.0000], [0,149.2751,-0.0104,0],[0,261.6091,-0.0103,0]]).reshape((4,4))
    # B = np.array([ 0,0,49.7275,49.1493]).reshape((4,1))

    return A, B


def LQR_control():
    # Cost matrices for LQR
    Q = np.diag(np.array([1, 1, 1, 1]))  # state_dimension = 4
    R = np.eye(1)  # control_dimension = 1

    A, B = computeAB(np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.0]))

    # Use if discrete forward dynamics is used
    X = sp_linalg.solve_discrete_are(A, B, Q, R)
    K = np.dot(np.linalg.pinv(R + np.dot(B.T, np.dot(X, B))), np.dot(B.T, np.dot(X, A)))

    # Use for continuous version of LQR
    # X = sp_linalg.solve_continuous_are(A, B, Q, R)
    # K = np.dot(np.linalg.pinv(R), np.dot(B.T, X))
    return np.squeeze(K, 0)


def main():
    """
    K obtained from dicrete dynamics + discrete LQR and continuous dynamics + continuous LQR should approximately match
    quanser workbook and more importantly achieve balance on the Qube Hardware
    """
    K_real = [-2.0, 35.0, -1.5, 3.0]  # Correct K from quanser workbook
    K_calc = LQR_control().tolist()
    print("The two following should be close to each other")
    print("\tThe gains from Quanser are:", K_real)
    print("\tThe calculated gains are:  ", K_calc)


if __name__ == "__main__":
    main()
