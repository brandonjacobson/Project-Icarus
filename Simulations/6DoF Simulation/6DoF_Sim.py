import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import scipy

matplotlib.use('TkAgg')

# Physical Parameters
m = 1  # Mass (kg)
L = 0.30  # Arm Length (m)
Ixx, Iyy, Izz = 0.01, 0.01, 0.02  # Moments of Inertia (kg*m^2)
g = -9.81  # Gravity (m/s^2)

# Motor Parameters
k_thrust = 1e-6  # Thrust Coefficient
k_torque = 1.5e-8  # Torque Coefficient
motor_gain = 15.0  # Motor Response Speed
rpm_max = 6000  # Maximum Motor RPM

# Rotor Config
rotors = [
    [1, 1, 'CCW'],
    [-1, -1, 'CCW'],
    [1, -1, 'CW'],
    [-1, 1, 'CW']
]

# Sensor Parameters
# Altimeter
alt_noise_std = 0.00  # Altimeter Noise Std Dev (5 cm)
alt_update_rate = 50.0  # Altimeter Update Rate (50 Hz)
alt_timer = 0.0  # Used to add sensor noise in loop
# GPS
gps_noise_std = 0.00  # GPS Noise Std Dev
gps_update_rate = 50.0  # GPS Update Rate ( Hz)
gps_timer = 0.0  # Used to add GPS noise in loop
# Gyro
gyro_noise_std = 0.00  # Gyro Noise Std Dev
gyro_update_rate = 50.0  # Gyro Update Rate ( Hz)
gyro_timer = 0.0  # Used to add Gyro noise in loop

# Control Parameters
# Position PID Gains
kp_x, ki_x, kd_x = 1.0, 0.0, 0.8  # X Position Gains
kp_y, ki_y, kd_y = 1.0, 0.0, 0.8  # Y Position Gains (1.3, 0.00, 0.5)
kp_z, ki_z, kd_z = 4.0, 0.0, 3.0  # Z Position Gains (4.0, 0.0, 3.0)
# Attitude PID Gains
kp_phi, ki_phi, kd_phi = 0.07, 0.0, 0.20  # Roll Gain (0.07, 0.0, 0.20)
kp_theta, ki_theta, kd_theta = 0.07, 0.0, 0.20  # Pitch Gain
kp_psi, ki_psi, kd_psi = 0.5, 0.0, 0.1  # Yaw Gain
# Filter Alpha
alpha = 0.6

# PID State
integral_x, prev_error_x, derivative_prev_x = 0.0, 0.0, 0.0
integral_y, prev_error_y, derivative_prev_y = 0.0, 0.0, 0.0
integral_z, prev_error_z, derivative_prev_z = 0.0, 0.0, 0.0
integral_phi, prev_error_phi, derivative_prev_phi = 0.0, 0.0, 0.0
integral_theta, prev_error_theta, derivative_prev_theta = 0.0, 0.0, 0.0
integral_psi, prev_error_psi, derivative_prev_psi = 0.0, 0.0, 0.0

# Simulation Parameters
dt = 0.01  # Timestep (s)
t_final = 30  # Simulation Final Time (s)
steps = int(t_final / dt)  # Number of Time Steps

# Initial State
# Position
x, y, z = 0.0, 0.0, 0.0
# Velocity
vx, vy, vz = 0.0, 0.0, 0.0
# Attitude (Euler Angles)
phi, theta, psi = 0.0, 0.0, 0.0
p_rate, q_rate, r_rate = 0.0, 0.0, 0.0
# Motor States
rpm1, rpm2, rpm3, rpm4 = 0.0, 0.0, 0.0, 0.0
# Measured States
x_meas, y_meas, z_meas = 0.0, 0.0, 0.0
phi_meas, theta_meas, psi_meas = 0.0, 0.0, 0.0
p_meas, q_meas, r_meas = 0.0, 0.0, 0.0

# Setpoints
x_target, y_target, z_target = 0.0, 3.0, 3.0
psi_target = 0.0

# Logging
time_log = []
x_log, y_log, z_log = [], [], []
vx_log, vy_log, vz_log = [], [], []
phi_log, theta_log, psi_log = [], [], []
p_log, q_log, r_log = [], [], []
rpm1_log, rpm2_log, rpm3_log, rpm4_log = [], [], [], []
thrust_log = []
x_target_log, y_target_log, z_target_log = [], [], []


# ------------------------- Functions ---------------------------------
def waypoints(time):
    if time >= 15:
        x_targ, y_targ, z_targ = 2.5, -2.5, 2.5
    elif time >= 7.0:
        x_targ, y_targ, z_targ = 5.0, 5.0, 5.0
    elif time >= 4.0:
        x_targ, y_targ, z_targ = 1.0, 1.0, 1.0
    else:
        x_targ, y_targ, z_targ = 0.0, 0.0, 0.0
    return x_targ, y_targ, z_targ

def rotation_matrix(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    R = np.array([
        [cpsi*cth,  cpsi*sth*sphi - spsi*cphi,  cpsi*sth*cphi + spsi*sphi],
        [spsi*cth,  spsi*sth*sphi + cpsi*cphi,  spsi*sth*cphi - cpsi*sphi],
        [   -sth ,              cth*sphi,                   cth*cphi]
    ])
    return R



def body_to_euler_rates(phi, theta, p, q, r):
    phi_dot = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
    theta_dot = q * np.cos(phi) - r * np.sin(phi)
    psi_dot = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)
    return np.array([phi_dot, theta_dot, psi_dot])


def PID(setpoint, pv, kp, ki, kd, prev_error, integral, dt, derivative_prev, alpha):
    error = setpoint - pv
    integral += error * dt
    derivative_raw = (error - prev_error) / dt
    derivative_filtered = alpha * derivative_prev + (1 - alpha) * derivative_raw
    control = kp * error + ki * integral + kd * derivative_filtered
    #integral = np.clip(integral, -100.0, 100.0)
    return control, error, integral, derivative_filtered


def mixer_matrix(rotors, gamma, L):  # Cookie Robotics Solution
    A = np.array([
        [1] * len(rotors),
        [-i[1] * (L / np.sqrt(2)) for i in rotors],
        [i[0] * (L / np.sqrt(2)) for i in rotors],
        [-gamma if i[2] == 'CW' else gamma for i in rotors]
    ])
    A = np.linalg.pinv(A)
    return A  # A is a normalized matrix of motor values, each row is a motor and each column is the inputs from thrust, roll, pitch, and yaw to be mixed.


def mixer(mixer_matrix, total_thrust, tau_phi, tau_theta, tau_psi):
    u = np.array([total_thrust, tau_phi, tau_theta, tau_psi])
    motor_forces = mixer_matrix @ u
    return motor_forces


# --------------------- Main Controls ------------------------
gamma = k_torque / k_thrust
B_mixer = mixer_matrix(rotors, gamma, L)

# Main Control Loop - MATLAB Drone Simulation and Control Solution
for i in range(steps):
    time = i * dt
    # Sensor Updates
    # Altimeter Update
    alt_timer += dt
    if alt_timer >= 1.0 / alt_update_rate:
        alt_timer = 0.0
        z_meas = z + np.random.normal(0.0, alt_noise_std)
    # GPS Update
    gps_timer += dt
    if gps_timer >= 1.0 / gps_update_rate:
        gps_timer = 0.0
        x_meas = x + np.random.normal(0.0, gps_noise_std)
        y_meas = y + np.random.normal(0.0, gps_noise_std)
    # Gyro Update
    gyro_timer += dt
    if gyro_timer >= 1.0 / gyro_update_rate:
        gyro_timer = 0.0
        p_meas = p_rate + np.random.normal(0.0, gyro_noise_std)
        q_meas = q_rate + np.random.normal(0.0, gyro_noise_std)
        r_meas = r_rate + np.random.normal(0.0, gyro_noise_std)
        # Here we assume that gyro is directly giving Euler Angles which is unrealistic,
        # but useful for primary testing. In the future it will be replaced by gyro and accel KF.
        phi_meas = phi #+ np.random.normal(0.0, 0.01)
        theta_meas = theta #+ np.random.normal(0.0, 0.01)
        psi_meas = psi #+ np.random.normal(0.0, 0.01)

    # Position Control (Outer Loop)
    # Z-axis Control
    thrust_feedback, prev_error_z, integral_z, derivative_prev_z = PID(z_target, z_meas, kp_z, ki_z, kd_z, prev_error_z,
                                                                       integral_z, dt, derivative_prev_z, alpha)
    # Feedforward + Feedback
    total_thrust = m * abs(g) + thrust_feedback
    total_thrust = np.clip(total_thrust, 0.0, 15.0)
    # X-axis Control -> Desired Pitch Angle
    x_cmd, prev_error_x, integral_x, derivative_prev_x = PID(x_target, x_meas, kp_x, ki_x, kd_x, prev_error_x,
                                                             integral_x, dt, derivative_prev_x, alpha)
    theta_desired = x_cmd
    theta_desired = np.clip(theta_desired, np.radians(-25), np.radians(25))
    # Y-axis Control -> Desired Roll Angle
    y_cmd, prev_error_y, integral_y, derivative_prev_y = PID(y_target, y_meas, kp_y, ki_y, kd_y, prev_error_y,
                                                             integral_y, dt, derivative_prev_y, alpha)
    phi_desired = -y_cmd
    phi_desired = np.clip(phi_desired, np.radians(-25), np.radians(25))
    # # Yaw Setpoint
    psi_desired = 0.0  # psi_target

    # Attitude Control (Inner Loop)
    # Roll Control (phi) -> Roll moment
    tau_phi_cmd, prev_error_phi, integral_phi, derivative_prev_phi = PID(phi_desired, phi_meas, kp_phi, ki_phi, kd_phi,
                                                                         prev_error_phi, integral_phi, dt,
                                                                         derivative_prev_phi, alpha)
    # Pitch Control (theta) -> Pitch moment
    tau_theta_cmd, prev_error_theta, integral_theta, derivative_prev_theta = PID(theta_desired, theta_meas, kp_theta,
                                                                                 ki_theta, kd_theta, prev_error_theta,
                                                                                 integral_theta, dt,
                                                                                 derivative_prev_theta, alpha)
    # Yaw Control (psi) -> Yaw moment
    tau_psi_cmd, prev_error_psi, integral_psi, derivative_prev_psi = PID(psi_desired, psi_meas, kp_psi, ki_psi, kd_psi,
                                                                         prev_error_psi, integral_psi, dt,
                                                                         derivative_prev_psi, alpha)

    # Control Allocation (Mixer)
    motor_forces = mixer(B_mixer, total_thrust, tau_phi_cmd, tau_theta_cmd, tau_psi_cmd)
    F1_cmd = motor_forces[0]
    F2_cmd = motor_forces[1]
    F3_cmd = motor_forces[2]
    F4_cmd = motor_forces[3]
    # Motor Saturation
    max_thrust_per_motor = (rpm_max ** 2) * k_thrust
    F1_cmd = np.clip(F1_cmd, 0.0, max_thrust_per_motor)
    F2_cmd = np.clip(F2_cmd, 0.0, max_thrust_per_motor)
    F3_cmd = np.clip(F3_cmd, 0.0, max_thrust_per_motor)
    F4_cmd = np.clip(F4_cmd, 0.0, max_thrust_per_motor)

    # Simulation
    # Motor Dynamics (Converst thrust commands to RPM with motor lag)
    # Motor 1
    if F1_cmd > 0:
        rpm1_setpoint = np.sqrt(F1_cmd / k_thrust)
    else:
        rpm1_setpoint = 0.0
    rpm1 += motor_gain * (rpm1_setpoint - rpm1) * dt  # First-order lag
    rpm1 = np.clip(rpm1, 0, rpm_max)  # Saturate
    # Motor 2
    if F2_cmd > 0:
        rpm2_setpoint = np.sqrt(F2_cmd / k_thrust)
    else:
        rpm2_setpoint = 0.0
    rpm2 += motor_gain * (rpm2_setpoint - rpm2) * dt
    rpm2 = np.clip(rpm2, 0, rpm_max)
    # Motor 3
    if F3_cmd > 0:
        rpm3_setpoint = np.sqrt(F3_cmd / k_thrust)
    else:
        rpm3_setpoint = 0.0
    rpm3 += motor_gain * (rpm3_setpoint - rpm3) * dt
    rpm3 = np.clip(rpm3, 0, rpm_max)
    # Motor 4
    if F4_cmd > 0:
        rpm4_setpoint = np.sqrt(F4_cmd / k_thrust)
    else:
        rpm4_setpoint = 0.0
    rpm4 += motor_gain * (rpm4_setpoint - rpm4) * dt
    rpm4 = np.clip(rpm4, 0, rpm_max)

    # Thrust and Torques from RPM
    # Individual Motor Thrusts (based on current RPM, not commands)
    F1 = k_thrust * rpm1 ** 2
    F2 = k_thrust * rpm2 ** 2
    F3 = k_thrust * rpm3 ** 2
    F4 = k_thrust * rpm4 ** 2
    # Total thrust in body frame
    Fz_body = F1 + F2 + F3 + F4
    F_body = np.array([0, 0, Fz_body])
    # Moments
    L_eff = L / np.sqrt(2)
    gamma = k_torque / k_thrust  # already defined above

    forces = [F1, F2, F3, F4]

    tau_phi_actual = 0.0
    tau_theta_actual = 0.0
    tau_psi_actual = 0.0

    for (x_sign, y_sign, direction), Fi in zip(rotors, forces):
        # Roll torque (about body x): lever arm in ±y
        tau_phi_actual += (-y_sign * L_eff) * Fi

        # Pitch torque (about body y): lever arm in ±x
        tau_theta_actual += (x_sign * L_eff) * Fi

        # Yaw torque from rotor reaction torque
        if direction == 'CW':
            tau_psi_actual += -gamma * Fi  # CW gives negative yaw
        else:
            tau_psi_actual += gamma * Fi  # CCW gives positive yaw

    # Disturbances
    # F_body += wind_force
    # tau_phi_actual += wind_torque_x

    # Translational Dynamics
    # Rotate thrust from body frame to inertial
    R = rotation_matrix(phi, theta, psi)
    F_world = R @ F_body
    # Accelerations in world frame
    ax = F_world[0] / m
    ay = F_world[1] / m
    az = F_world[2] / m + g
    # Integrate Velocity
    vx += ax * dt
    vy += ay * dt
    vz += az * dt
    # Integrate Position
    x += vx * dt
    y += vy * dt
    z += vz * dt

    # Rotational Dynamics
    # Angular Accel
    p_dot = (tau_phi_actual + (Iyy - Izz) * q_rate * r_rate) / Ixx
    q_dot = (tau_theta_actual + (Izz - Ixx) * r_rate * p_rate) / Iyy
    r_dot = (tau_psi_actual + (Ixx - Iyy) * p_rate * q_rate) / Izz
    # Integrate Angular rate
    p_rate += p_dot * dt
    q_rate += q_dot * dt
    r_rate += r_dot * dt
    # Convert body rates to Euler angle rates
    euler_rates = body_to_euler_rates(phi, theta, p_rate, q_rate, r_rate)
    phi_dot = euler_rates[0]
    theta_dot = euler_rates[1]
    psi_dot = euler_rates[2]
    # Integrate Euler angles
    phi += phi_dot * dt
    theta += theta_dot * dt
    psi += psi_dot * dt

    # Logging
    time_log.append(time)
    # Position Log
    x_log.append(x)
    y_log.append(y)
    z_log.append(z)
    # Velocity Log
    vx_log.append(vx)
    vy_log.append(vy)
    vz_log.append(vz)
    # Attitude Log
    phi_log.append(phi)
    theta_log.append(theta)
    psi_log.append(psi)
    # Angular Rate Log
    p_log.append(p_rate)
    q_log.append(q_rate)
    r_log.append(r_rate)
    # Motor RPMs Log
    rpm1_log.append(rpm1)
    rpm2_log.append(rpm2)
    rpm3_log.append(rpm3)
    rpm4_log.append(rpm4)
    # Total Thrust Log
    thrust_log.append(Fz_body)
    # Target Log for Waypoints
    x_target_log.append(x_target)
    y_target_log.append(y_target)
    z_target_log.append(z_target)

# ---------------------- Tuning Plots -----------------------

time = np.array(time_log)

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Position
axs[0].plot(time, x_log, label='x')
axs[0].plot(time, y_log, label='y')
axs[0].plot(time, z_log, label='z')
axs[0].plot(time, x_target_log, linestyle='--', label='x_target')
axs[0].plot(time, y_target_log, linestyle='--', label='y_target')
axs[0].plot(time, z_target_log, linestyle='--', label='z_target')
axs[0].set_ylabel('Position (m)')
axs[0].legend()
axs[0].grid(True)

# Attitude
axs[1].plot(time, np.degrees(phi_log), label='phi (deg)')
axs[1].plot(time, np.degrees(theta_log), label='theta (deg)')
axs[1].plot(time, np.degrees(psi_log), label='psi (deg)')
axs[1].set_ylabel('Angles (deg)')
axs[1].legend()
axs[1].grid(True)

# Motor RPMs
axs[2].plot(time, rpm1_log, label='rpm1')
axs[2].plot(time, rpm2_log, label='rpm2')
axs[2].plot(time, rpm3_log, label='rpm3')
axs[2].plot(time, rpm4_log, label='rpm4')
axs[2].set_ylabel('RPM')
axs[2].set_xlabel('Time (s)')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()



# ---------------------- Animation (3D Trajectory) -------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
motor_pos_body = np.array([
    [L / np.sqrt(2), L / np.sqrt(2), 0],  # Motor 1 (front-right)
    [-L / np.sqrt(2), -L / np.sqrt(2), 0],  # Motor 2 (back-left)
    [L / np.sqrt(2), -L / np.sqrt(2), 0],  # Motor 3 (front-left)
    [-L / np.sqrt(2), L / np.sqrt(2), 0],  # Motor 4 (back-right)
])


def update_frame(frame):
    ax.clear()

    # Get current state
    idx = frame * 5  # Skip fewer frames for smoother animation
    if idx >= len(x_log):
        idx = len(x_log) - 1

    pos = np.array([x_log[idx], y_log[idx], z_log[idx]])
    phi_curr = phi_log[idx]
    theta_curr = theta_log[idx]
    psi_curr = psi_log[idx]

    # Rotate motor positions to world frame
    R = rotation_matrix(phi_curr, theta_curr, psi_curr)
    motor_pos_world = (R @ motor_pos_body.T).T + pos

    # Draw quad arms as BLACK lines (4 arms from center to each motor)
    for i in range(4):
        ax.plot([pos[0], motor_pos_world[i, 0]],
                [pos[1], motor_pos_world[i, 1]],
                [pos[2], motor_pos_world[i, 2]], 'k-', linewidth=3)

    # Draw cross-bracing between motors for X-configuration
    ax.plot([motor_pos_world[0, 0], motor_pos_world[1, 0]],
            [motor_pos_world[0, 1], motor_pos_world[1, 1]],
            [motor_pos_world[0, 2], motor_pos_world[1, 2]], 'k-', linewidth=3, alpha=0.6)
    ax.plot([motor_pos_world[2, 0], motor_pos_world[3, 0]],
            [motor_pos_world[2, 1], motor_pos_world[3, 1]],
            [motor_pos_world[2, 2], motor_pos_world[3, 2]], 'k-', linewidth=3, alpha=0.6)

    # Draw motors as red dots with black edges
    #ax.scatter(motor_pos_world[:, 0], motor_pos_world[:, 1], motor_pos_world[:, 2],
    #           c='red', s=200, marker='o', edgecolors='black', linewidths=2)

    # Draw center as yellow dot
    #ax.scatter([pos[0]], [pos[1]], [pos[2]], c='yellow', s=200, marker='o',
    #           edgecolors='black', linewidths=2)

    # Draw trajectory trail
    trail_length = min(200, idx)
    if trail_length > 0:
        ax.plot(x_log[idx - trail_length:idx],
                y_log[idx - trail_length:idx],
                z_log[idx - trail_length:idx], 'g-', alpha=0.5, linewidth=2)

    # Draw target
    ax.scatter([x_target], [y_target], [z_target], marker='o',
               edgecolors='orange', linewidths=3)

    # Set labels and limits
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'Time: {time_log[idx]:.2f}s | Alt: {z_log[idx]:.2f}m | '
                 f'Phi: {np.degrees(phi_log[idx]):.1f}° | Theta: {np.degrees(theta_log[idx]):.1f}°',
                 fontsize=14)

    # Set consistent axis limits
    max_range = 5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range])

    # Add grid
    ax.grid(True, alpha=0.3)

    return ax,


# Create animation
num_frames = len(x_log) // 5
anim = animation.FuncAnimation(fig, update_frame, frames=num_frames,
                               interval=50, blit=False)

plt.show()

