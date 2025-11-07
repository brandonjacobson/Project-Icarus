import matplotlib
matplotlib.use("TkAgg")

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
m = 1.0
g = -9.8  # down negative
k_thrust = 1e-6

#Sensor Parameters
alt_noise_std = 0.05  #5 cm std dev
alt_update_rate = 50.0  #Freq of sensor (Hz)
alt_update_dt = 1.0/alt_update_rate
alt_timer = 0.0


#Thrust Constants
hover_thrust = -m * g #About 9.8 N
rpm_hover = np.sqrt(hover_thrust / k_thrust) #About 3130 rpm

#PID Gains
kp = 2.2  #Proportional (2.2)
ki = 0.06  #Integral (0)
kd = 2.7  #Derivative (2.7)

#PID State
integral = 0.0
prev_error = 0.0
derivative_prev = 0.0  #previous filtered derivative
alpha = 0.9  #low-pass factor (0-1) Higher = Smoother Deriv.

#Sim Settings
dt = 0.01  #Time step
t_final = 60.0
steps = int(t_final/dt)

#TARGET
setpoint = 10.0  #Target altitude (m)

#Initial Dynamics State
p = 0.0  #Altitude = pe^3
v = 0.0  #Velocity = ve^3
rpm = 0.0
p_meas = p  #Initialize measured altitude

#Logging
time_log = []
alt_log = []
rpm_log = []
thrust_log = []

#PID Controller
def PID(setpoint, pv, kp, ki, kd, previous_error, integral, dt, derivative_prev, alpha, thrust_cmd_unsat=None, thrust_cmd_sat=None):
    error = setpoint - pv
    integral += error * dt
    derivative_raw = (error - previous_error) / dt
    derivative_filtered = alpha * derivative_prev + (1 - alpha) * derivative_raw #Low-Pass Filter
    control = kp * error + ki * integral + kd * derivative_filtered
    #Anti-Windup (Clamping)
    integral = np.clip(integral, -20.0, 20.0)
    return control, error, integral, derivative_filtered

#Main Control Loop
for i in range(steps):
    time = i * dt

    #Disturbance Testing
    disturbance = 0.0
    #if 20.0 <= time <= 22.0:
     #   disturbance = -2.0 #Downward wind force (2 N)
    #if time >= 20.0:
    #    disturbance = -1.0 #Constand Downward wind (1 N)

    #Step Testing
    #if 20.0 <= time <= 40.0:
    #    setpoint = 15.0
    #if time > 40.0:
    #    setpoint = 7.0

    alt_timer += dt
    if alt_timer >= alt_update_dt:
        alt_timer = 0.0
        p_meas = p + np.random.normal(0.0, alt_noise_std)

    #PID Control
    thrust_cmd, error, integral, derivative_prev = PID(setpoint, p_meas, kp, ki, kd, prev_error, integral, dt, derivative_prev, alpha)
    prev_error = error

    #Thrust Saturation for Integral Windup
    thrust_cmd_unsat = hover_thrust + thrust_cmd
    thrust_cmd = np.clip(thrust_cmd_unsat, 0.0, 15.0) #clamp between 0 N and 15 N max thrust

    #Anti-Windup
    if thrust_cmd != thrust_cmd_unsat:
        integral -= error * dt #undo last integration step if saturated

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

##Performance Calculations
setpoint_arr = np.full_like(time_log, setpoint) #Make a constant log of target point
alt_arr = np.array(alt_log)
time_arr = np.array(time_log)

target = setpoint
steady_tolerance = 0.01 * target
target_indices = np.where(np.abs(alt_arr - target) <= steady_tolerance)[0]

#Overshoot
peak = alt_arr.max()
overshoot_pct = max(0, (peak-target) / target * 100)

#Settling Time (Calculated as first time is stays within +- 1% for the rest)
settling_time = None
for j in range(len(alt_arr)):
    window = alt_arr[j:]
    if np.all(np.abs(window-target) <= steady_tolerance):
        settling_time = time_arr[j]
        break

#Steady State Error (Average of last second)
steady_mask = time_arr >= (time_arr[-1] - 1.0)
steady_state_error = np.mean(alt_arr[steady_mask]) - target

print(f"Overshoot: {overshoot_pct:.1f}%")
#print(f"Settling time: {settling_time:.2f} s")
print(f"Steady-state Error: {steady_state_error:.3f} m")
print(f"Thrust_unsat={thrust_cmd_unsat:.2f}, thrust_sat={thrust_cmd:.2f}")

plt.plot(time_log, alt_log)
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.title("Noisy Low-Pass PID Control to 10m Hover")
plt.grid(True)
plt.show()

plt.plot(time_log, rpm_log)
plt.xlabel("Time (s)")
plt.ylabel("Motor RPM")
plt.title("Noisy Low-Pass RPM Graph for 10m Hover w/ PID")
plt.grid(True)
plt.show()