import matplotlib
matplotlib.use("TkAgg")

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
m = 1.0
g = -9.8  # down negative
k_thrust = 1e-6

#Thrust Constants
hover_thrust = -m * g #About 9.8 N
rpm_hover = np.sqrt(hover_thrust / k_thrust) #About 3130 rpm

#PID Gains
kp = 2.2  #Proportional (3)
ki = 0.0  #Integral (0.5)
kd = 2.7  #Derivative (2)

#PID State
integral = 0.0
prev_error = 0.0

#Sim Settings
dt = 0.01  #Time step
t_final = 60.0
steps = int(t_final/dt)

#TARGET
setpoint = 10.0 #Target altitude (m)

#Initial Dynamics State
p = 0.0 #Altitude = pe^3
v = 0.0 #Velocity = ve^3
rpm = 0.0

#Logging
time_log = []
alt_log = []
rpm_log = []
thrust_log = []

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

    #Disturbance Testing
    disturbance = 0.0
    #if 20.0 <= time <= 22.0:
     #   disturbance = -2.0 #Downward wind force (2 N)
    #if time >= 30.0:
     #   disturbance = -1.0 #Constand Downward wind (1 N)

    #Step Testing
    if 20.0 <= time <= 40.0:
        setpoint = 15.0
    if time > 40.0:
        setpoint = 7.0

    #PID Control
    thrust_cmd, error, integral = PID(setpoint, p, kp, ki, kd, prev_error, integral, dt)
    prev_error = error

    #Total Thrust
    thrust_cmd = hover_thrust + thrust_cmd
    thrust_cmd = max(0.0, thrust_cmd)

    #Convert Thrust to RPM
    if thrust_cmd > 0:
        rpm_setpoint = np.sqrt(thrust_cmd / k_thrust)
    else:
        rpm_setpoint = 0.0

    #Motor Model Parameters
    motor_gain = 15.0  #Higher = Faster Response

    #Motor Response (1st-order Lag)
    rpm += motor_gain * (rpm_setpoint - rpm) *dt

    #Motor RPM Saturation
    rpm = np.clip(rpm, 0, 6000)

    #Convert motor rpm to thrust
    thrust = k_thrust * rpm**2

    #EOMs
    a = (thrust+disturbance)/m + g
    v += a*dt
    p += v*dt

    #Log Data
    time_log.append(time)
    alt_log.append(p)
    rpm_log.append(rpm)
    thrust_log.append(thrust)

plt.plot(time_log, alt_log)
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.title("PID Control to 10m Hover")
plt.grid(True)
plt.show()

plt.plot(time_log, rpm_log)
plt.xlabel("Time (s)")
plt.ylabel("Motor RPM")
plt.title("RPM Graph for 10m Hover w/ PID")
plt.grid(True)
plt.show()