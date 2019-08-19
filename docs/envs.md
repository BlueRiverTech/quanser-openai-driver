
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
- Episode ends once the theta angle is greater than 90 degrees.
- Reward is a function of the angles theta (arm angle) and alpha (pendulum), and the alpha angular velocity.
    - Encourages the the arm to stay centered, the pendulum to stay upright, and to stay stationary.


### QubeBeginUprightEnv
Info:

- Reset starts the pendulum from the top (flipped up/inverted).
- The task is to hold the pendulum upright.
- Episode ends once the alpha angle is greater the 20 degrees or theta angle is greater than 90 degrees.
- Reward is a function of the angles theta (arm angle) and alpha (pendulum), and the alpha angular velocity.
    - Encourages the the arm to stay centered, the pendulum to stay upright, and to stay stationary.
