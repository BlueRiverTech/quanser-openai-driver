from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from scipy.integrate import odeint
import math

# Motor
Rm = 8.4  # Resistance
kt = 0.042  # Current-torque (N-m/A)
km = 0.042  # Back-emf constant (V-s/rad)

# Rotary Arm
mr = 0.095  # Mass (kg)
Lr = 0.085  # Total length (m)
Jr = mr * Lr**2 / 12  # Moment of inertia about pivot (kg-m^2)

# Pendulum Link
mp = 0.024  # Mass (kg)
Lp = 0.129  # Total length (m)
Jp = mp * Lp**2 / 12  # Moment of inertia about pivot (kg-m^2)

Br = Dr = 0.0005  # Equivalent viscous damping coefficient (N-m-s/rad)
Bp = Dp = 0.00005  # Equivalent viscous damping coefficient (N-m-s/rad)
g = 9.81  # Gravity constant

def diff_forward_model(state, t, action, dt):
    theta = state[0]; alpha = state[1]; theta_dot = state[2]; alpha_dot = state[3]; Vm = action

    tau = (km * (Vm - km * theta_dot)) / Rm  # torque
    tau = -1.0*tau #Hypothesis, please remove if not found true
    # Solved by Kirill - Cazzolato and Prime
    '''
    theta_dot_dot = (-Lp*Lr*mp*(8.0*Bp*alpha_dot - 4.0*Jp*theta_dot**2*math.sin(2.0*alpha) - Lp**2*mp*theta_dot**2*math.sin(2.0*alpha)\
     + 4.0*Lp*g*mp*math.sin(alpha))*math.cos(alpha) + (4.0*Jp + Lp**2*mp)*(4.0*Br*theta_dot + \
     4.0*Jp*alpha_dot*theta_dot*math.sin(2.0*alpha) + Lp**2*mp*alpha_dot*theta_dot*math.sin(2.0*alpha)\
      - 2.0*Lp*Lr*mp*alpha_dot**2*math.sin(alpha) - 4.0*tau))/(4.0*Lp**2*Lr**2*mp**2*math.cos(alpha)**2 -\
       (4.0*Jp + Lp**2*mp)*(4.0*Jp*math.sin(alpha)**2 + 4.0*Jr + Lp**2*mp*math.sin(alpha)**2 + 4.0*Lr**2*mp + Lr**2*mr))
    alpha_dot_dot = 0.5*(-4.0*Lp*Lr*mp*(4.0*Br*theta_dot + 4.0*Jp*alpha_dot*theta_dot*math.sin(2.0*alpha) + \
        Lp**2*mp*alpha_dot*theta_dot*math.sin(2.0*alpha) - 2.0*Lp*Lr*mp*alpha_dot**2*math.sin(alpha) - 4.0*tau)*math.cos(alpha) \
        + (8.0*Bp*alpha_dot - 4.0*Jp*theta_dot**2*math.sin(2.0*alpha) - Lp**2*mp*theta_dot**2*math.sin(2.0*alpha) + \
            4.0*Lp*g*mp*math.sin(alpha))*(4.0*Jp*math.sin(alpha)**2 + 4.0*Jr + Lp**2*mp*math.sin(alpha)**2 + \
            4.0*Lr**2*mp + Lr**2*mr))/(4.0*Lp**2*Lr**2*mp**2*math.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jp*math.sin(alpha)**2 +\
             4.0*Jr + Lp**2*mp*math.sin(alpha)**2 + 4.0*Lr**2*mp + Lr**2*mr))
    '''
    # From Rotary Pendulum Workbook
    
    alpha_dot_dot = (2.0*Lp*Lr*mp*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*np.sin(2.0*alpha) + \
    2.0*Lp*Lr*alpha_dot**2*mp*np.sin(alpha) - 4.0*tau)*np.cos(alpha) - 0.5*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + \
    4.0*Lr**2*mp)*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha)))\
    /(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp))

    theta_dot_dot = (-Lp*Lr*mp*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha))*np.cos(alpha) + \
    (4.0*Jp + Lp**2*mp)*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*np.sin(2.0*alpha) + 2.0*Lp*Lr*alpha_dot**2*mp*np.sin(alpha)\
    - 4.0*tau))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp))
        

    diff_state = np.array([theta_dot,alpha_dot,theta_dot_dot,alpha_dot_dot]).reshape((4,))
    diff_state = np.array(diff_state,dtype='float64')
    return diff_state

def forward_model(theta, alpha, alpha_unnorm, theta_dot, alpha_dot, Vm, dt, euler_steps = 25):
    t = np.linspace(0.,dt,2)

    state = np.array([theta,alpha,theta_dot,alpha_dot])
    next_state = np.array(odeint(diff_forward_model,state,t,args=(Vm,dt)))[1,:]

    theta = next_state[0]; alpha = next_state[1]; theta_dot = next_state[2]; alpha_dot = next_state[3]
    

    theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
    alpha = ((alpha + np.pi) % (2 * np.pi)) - np.pi
    # Ensures that the encoder value is the same as in the Qube
    alpha_unnorm = next_state[1] 

    return theta, alpha, alpha_unnorm, theta_dot, alpha_dot 

# Use numba if installed
try:
    import numba
    forward_model = numba.jit(forward_model)
except:
    print('Warning: Install \'numba\' for faster simulation.')


class QubeServo2Simulator(object):
    '''Simulator that has the same interface as the hardware wrapper.'''
    def __init__(self,
                 safe_operating_voltage=18.0,
                 euler_steps=1,
                 frequency=1000):
        self._time_step = 1.0 / frequency
        self._euler_steps = euler_steps
        self.state = [0, 0, 0, 0, 0]

    def __enter__(self):
        self.state = (np.random.randn(5) * 0.0001).tolist()
        return self

    def __exit__(self, type, value, traceback):
        pass

    def reset_up(self):
        self.state = (np.random.randn(5) * 0.0001 + np.array([0, 0, 0, 0, 0])).tolist()
        
    def reset_down(self):
        self.state = (np.random.randn(5) * 0.0001 + np.array([0, np.pi, np.pi, 0, 0])).tolist()

    def reset_encoders(self):
        pass
 
    def action(self, action):
        self.state = forward_model(*self.state,action,self._time_step,self._euler_steps)
        # Convert the angles into counts ()
        theta_counts = int((self.state[0] / np.pi) * 2048)
        alpha_counts = int((self.state[2] / np.pi) * 2048)  # use unnormalized alpha
        encoders = [theta_counts, alpha_counts]
        currents = [action / 8.4]  # 8.4 is resistance
        others = [0.]  #[tach0, other_stuff]

        return currents, encoders, others

