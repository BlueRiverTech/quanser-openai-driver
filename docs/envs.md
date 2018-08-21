
# Environments
Quanser OpenAI Driver currently supports two pieces of hardware: the Qube Servo 2 USB and the Quanser Aero.

# Qube
For more information about the Qube click [here](https://www.quanser.com/products/qube-servo-2/)

### QubeBaseEnv
The base class for Qube environments.
Info:

- Reset starts the pendulum from the bottom (at rest).
- Has no reward function.


### QubeBeginDownEnv
Info:

- Reset starts the pendulum from the bottom (at rest).
- The task is to flip up the pendulum and hold it upright.
- Reward is a function of the angles theta (arm angle) and alpha (pendulum), and the alpha angular velocity.
    - Encourages the the arm to stay centered, the pendulum to stay upright, and to stay stationary.


### QubeBeginUprightEnv
Info:

- Reset starts the pendulum from the top (flipped up/inverted).
- The task is to hold the pendulum upright.
- Episode ends once the angle is outside the tolerance angle (falls too much).
- Reward is a function of the angles theta (arm angle) and alpha (pendulum), and the alpha angular velocity.
    - Encourages the the arm to stay centered, the pendulum to stay upright, and to stay stationary.


# Aero
For more information about the Aero click [here](https://www.quanser.com/products/quanser-aero/)

### AeroEnv
The main Aero environment.
Info:

- Reset applies no action (0 voltages) to the motors for a set number of timesteps.
- Reward is a function of the pitch and yaw compared to the reference pitch and yaw (the pitch and yaw at the start of the episode), and the velocities and accelerations in the x, y and z directions.
