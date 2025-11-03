import matplotlib
matplotlib.use("TkAgg")

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
m = 1.0
g = -9.8  # down negative

#PID Gains
kp = 5.0  #Proportional (3)
ki = 0.6  #Integral (0.5)
kd = 4.0  #Derivative (2)

#PID State
integral = 0.0
prev_error = 0.0

#Sim Settings
dt = 0.01  #Time step
t_final = 60.0
steps = int(t_final/dt)

#TARGET
setpoint = 20.0 #Try to reach exactly 10m hover

#Initial Dynamics State
p = 0.0 #Altitude = pe^3
v = 0.0 #Velocity = ve^3

#Logging
time_log = []
alt_log = []

#PID Controller
def PID(setpoint, pv, kp, ki, kd, previous_error, integral, dt):
    error = setpoint - pv
    integral += error * dt
    derivative = (error - previous_error) / dt
    control = kp * error + ki * integral + kd * derivative
    return control, error, integral

#Main Control Loop
for i in range(steps):
    time = i * dt

    #PID Control
    thrust_cmd, error, integral = PID(setpoint, p, kp, ki, kd, prev_error, integral, dt)
    prev_error = error

    #Thrust Saturation
    thrust = np.clip(thrust_cmd, 0, 20)

    #EOMs
    a = thrust/m + g
    v += a*dt
    p += v*dt

    #Log Data
    time_log.append(time)
    alt_log.append(p)

plt.plot(time_log, alt_log)
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.title("PID Control to 10m Hover")
plt.grid(True)
plt.show()