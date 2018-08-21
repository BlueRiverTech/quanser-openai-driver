
# Quanser
The OpenAI Gym environments use one of the following to calculate each step.

Quanser wrapper is a hardware wrapper that allows the Qube Servo 2 and Quanser Aero to be used directly in OpenAI Gym.
Quanser Simulator simulates the Qube Servo 2 only (Aero is currently not supported for simulation).

# Quanser Wrapper
The implementation of a Python wrapper around Quanser's C-based HIL SDK.
This is written in Cython.

All changes to this are recompiled if you install the gym_brt package with Cython installed.

# Quanser Simulator
A simulator and wrapper to use a simulator directly in the Qube environments instead of using the real hardware.
This is useful for transfer learning from simulation to real hardware and for people who don't have access to the Qube Servo 2.
