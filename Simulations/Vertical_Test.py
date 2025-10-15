import matplotlib
matplotlib.use("TkAgg")

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
m = 1.0
g = -9.8  # down negative
T_hover = -m * g          # cancels gravity (positive)
T_full  = 1.5 * T_hover   # stronger upward thrust
T_cut   = 0.0

# Dynamics helpers
def dyn_with_thrust(t, y, T):
    p3, p6 = y
    dp3_dt = p6
    dp6_dt = T / m + g
    return [dp3_dt, dp6_dt]

def dyn_ascend(t, y):
    return dyn_with_thrust(t, y, T_full)

def dyn_hover(t, y):
    return dyn_with_thrust(t, y, T_hover)

def dyn_cut(t, y):
    return dyn_with_thrust(t, y, T_cut)

# Events
def reach_target_event(t, y):
    p3, p6 = y
    return p3 - 10.0
reach_target_event.terminal = True
reach_target_event.direction = +1

def hit_ground_event(t, y):
    p3, p6 = y
    return p3
hit_ground_event.terminal = True
hit_ground_event.direction = -1

# 1) Ascend to 10 m
y0 = [0.0, 0.0]  # [p3, p6]
sol1 = solve_ivp(dyn_ascend, (0.0, 60.0), y0, events=reach_target_event, max_step=0.05)
t1 = sol1.t_events[0][0]
p3_1, p6_1 = sol1.y_events[0][0]

# 2) Hover for 5 s
sol2 = solve_ivp(dyn_hover, (t1, t1 + 5.0), [p3_1, p6_1], max_step=0.05)
p3_2, p6_2 = sol2.y[0, -1], sol2.y[1, -1]

# 3) Cut thrust and descend to ground
sol3 = solve_ivp(dyn_cut, (sol2.t[-1], sol2.t[-1] + 120.0), [p3_2, p6_2], events=hit_ground_event, max_step=0.05)

# Concatenate and plot p3 vs time
t = np.concatenate([sol1.t, sol2.t[1:], sol3.t[1:]])
p3 = np.concatenate([sol1.y[0], sol2.y[0, 1:], sol3.y[0, 1:]])

plt.figure()
plt.plot(t, p3, linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("p3 (vertical position, m)")
plt.title("Ascend → Hover (5 s) → Free fall landing")
plt.grid(True)
plt.show()
