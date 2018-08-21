
# Controllers

### Control
Control base class.

### NoControl
Applies no action (sets voltages to zero).

### RandomControl
Applies a random action (samples from the action space).

### AeroControl
Uses PID control to minimize (pitch - reference pitch) + (yaw - reference yaw), where reference is the original position.

### QubeFlipUpControl
Uses a mixed mode controller that uses gains found from LQR to do the flip up when the pendulum angle is over than 20 degrees off upright, and uses PID control and filtering to hold the pendulum upright when under 20 degrees.

### QubeHoldControl
Holding control uses PID with filtering, and outside of 20 degrees use no control.
