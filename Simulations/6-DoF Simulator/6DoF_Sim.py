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
# Accelerometer (body-frame specific force — gravity vector + linear accel + noise)
accel_noise_std  = 0.05    # m/s² per axis (typical MEMS IMU e.g. ICM-20689)
# Altimeter (barometer / sonar)
alt_noise_std    = 0.01    # m  — 1 cm std dev (realistic for 50 Hz barometer/sonar)
alt_update_rate  = 50.0    # Hz
alt_timer        = 0.0
# GPS (horizontal position, slow and noisy)
gps_noise_std    = 0.1    # m  — reduced for CF testing (realistic GPS needs position KF)
gps_update_rate  = 5.0     # Hz — GPS is much slower than IMU
gps_timer        = 0.0
R_gps = np.diag([gps_noise_std**2, gps_noise_std**2])  # GPS x,y measurement noise covariance (2D; z handled by altimeter)
# IMU (gyro + accel combined, runs every control timestep)
gyro_noise_std   = 0.003   # rad/s per axis per sample
gyro_bias_drift  = 0.0001  # rad/s/√s — random walk rate of gyro bias
gyro_update_rate = 100.0   # Hz — matches control loop rate
gyro_timer       = 0.0

# Control Parameters
# ---------------------------------------------------------------------------
# Gains are computed from the linearised plant at hover using pole placement.
# Cascade rule: inner (attitude) bandwidth must be 5-10x the outer (position).
#
# ATTITUDE inner loop — plant: phi_ddot = tau_phi / Ixx  (pure double integrator)
#   For a PD controller: kp = Ixx*wn^2,  kd = 2*zeta*wn*Ixx
#   Target: wn_att = 8 rad/s,  zeta = 0.7  → settling ~0.7 s
#   kp = 0.01 * 64  = 0.64
#   kd = 0.01 * 2 * 0.7 * 8 = 0.112
#
# POSITION XY outer loop — plant (inner loop idealised as fast): x_ddot ≈ g * theta
#   The position PID outputs theta_des, so effective plant is g/s^2.
#   kp = wn^2 / g,   kd = 2*zeta*wn / g
#   Target: wn_pos = 1.2 rad/s (< wn_att/5 = 1.6 rad/s), zeta = 0.9 → settling ~3.7 s
#   kp = 1.44 / 9.81 = 0.147
#   kd = 2 * 0.9 * 1.2 / 9.81 = 0.220
#
# POSITION Z — plant (feedforward cancels gravity): z_ddot = thrust_feedback / m
#   kp = m*wn^2,  kd = 2*zeta*wn*m
#   Target: wn_z = 2 rad/s, zeta = 0.8 → settling ~2.5 s
#   kp = 1 * 4 = 4.0,  kd = 2 * 0.8 * 2 * 1 = 3.2
#
# YAW inner loop — plant: psi_ddot = tau_psi / Izz
#   kp = Izz*wn^2,  kd = 2*zeta*wn*Izz
#   Target: wn_psi = 3 rad/s, zeta = 0.7 → settling ~1.9 s
#   kp = 0.02 * 9 = 0.18,  kd = 2 * 0.7 * 3 * 0.02 = 0.084
# ---------------------------------------------------------------------------

# Position PID Gains
kp_x, ki_x, kd_x = 0.147, 0.0, 0.220  # X  — wn=1.2 rad/s, zeta=0.9
kp_y, ki_y, kd_y = 0.147, 0.0, 0.220  # Y  — wn=1.2 rad/s, zeta=0.9
kp_z, ki_z, kd_z = 4.0,   0.0, 3.2    # Z  — wn=2.0 rad/s, zeta=0.8
# Attitude PID Gains
kp_phi,   ki_phi,   kd_phi   = 0.64, 0.0, 0.112  # Roll  — wn=8 rad/s, zeta=0.7
kp_theta, ki_theta, kd_theta = 0.64, 0.0, 0.112  # Pitch — wn=8 rad/s, zeta=0.7
kp_psi,   ki_psi,   kd_psi   = 0.18, 0.0, 0.084  # Yaw   — wn=3 rad/s, zeta=0.7
# Derivative Low-Pass Filter coefficients (IIR: y = alpha*y_prev + (1-alpha)*x_raw)
# alpha=0 → no filtering; alpha→1 → heavy smoothing (adds phase lag, causes instability)
# Approximate cutoff frequency: fc ≈ (1 - alpha) / (2 * pi * dt)
alpha_pos = 0.40  # fc ≈  6 Hz — outer position loop is slow, moderate filtering OK
alpha_att = 0.15  # fc ≈ 24 Hz — inner attitude loop is fast, keep derivative responsive
alpha_cf = 0.95  # CF blending factor — adjust for more/less gyro vs accel trust

# PID State
integral_x, prev_error_x, derivative_prev_x = 0.0, 0.0, 0.0
integral_y, prev_error_y, derivative_prev_y = 0.0, 0.0, 0.0
integral_z, prev_error_z, derivative_prev_z = 0.0, 0.0, 0.0
integral_phi, prev_error_phi, derivative_prev_phi = 0.0, 0.0, 0.0
integral_theta, prev_error_theta, derivative_prev_theta = 0.0, 0.0, 0.0
integral_psi, prev_error_psi, derivative_prev_psi = 0.0, 0.0, 0.0
# Previous measured values for derivative-on-measurement (one per loop)
prev_pv_x, prev_pv_y, prev_pv_z = 0.0, 0.0, 0.0
prev_pv_phi, prev_pv_theta, prev_pv_psi = 0.0, 0.0, 0.0

# Simulation Parameters
dt = 0.01  # Timestep (s)
t_final = 30  # Simulation Final Time (s)
steps = int(t_final / dt)  # Number of Time Steps

# Initial State
# Position
x, y, z = 0.0, 0.0, 0.0
# Velocity
vx, vy, vz = 0.0, 0.0, 0.0
# World-frame linear acceleration (previous step; used by accel model in IMU block)
ax, ay, az = 0.0, 0.0, 0.0
# Attitude (Euler Angles)
phi, theta, psi = 0.0, 0.0, 0.0
p_rate, q_rate, r_rate = 0.0, 0.0, 0.0
# Motor States
rpm1, rpm2, rpm3, rpm4 = 0.0, 0.0, 0.0, 0.0
# Measured States
x_meas, y_meas, z_meas = 0.0, 0.0, 0.0
vx_meas, vy_meas = 0.0, 0.0    # GPS-rate velocity estimates (m/s)
vz_meas = 0.0                  # altimeter-rate velocity estimate (m/s)
phi_meas, theta_meas, psi_meas = 0.0, 0.0, 0.0
p_meas, q_meas, r_meas = 0.0, 0.0, 0.0
ax_meas, ay_meas, az_meas = 0.0, 0.0, 0.0   # accelerometer (body frame, m/s²)
gyro_bias_p, gyro_bias_q, gyro_bias_r = 0.0, 0.0, 0.0  # gyro bias states (rad/s)
x_state = np.array([[x], [y], [z], [vx], [vy], [vz]])  # state vector
a_world_prev = np.zeros((3, 1))  # world-frame linear accel from previous KF step (for CF correction)

# Kalman Filter Covariance Matrices
P = np.eye(6) * 10  # Initial covariance (position and velocity)
sigma_gps_xy = 2.0 # m, GPS horizontal measurement noise std dev (TUNE)
sigma_gps_z = 3.0  # m, GPS vertical measurement noise std dev (TUNE)


# Waypoints — edit this list to define the mission
# Each entry is [x, y, z] in meters.
waypoints_list = [
    [0.0, 0.0, 2.0],   # WP0: takeoff / hover
    [3.0, 0.0, 2.0],   # WP1
    [3.0, 3.0, 3.0],   # WP2
    [0.0, 3.0, 3.0],   # WP3
    [0.0, 0.0, 2.0],   # WP4: return home
]
wp_accept_radius = 0.35   # m — advance to next WP when within this distance
wp_index = 0
x_target, y_target, z_target = waypoints_list[0]
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
wp_index_log = []
# Kalman filter vs truth logging
x_kf_log, y_kf_log, z_kf_log = [], [], []
x_gps_log, y_gps_log, t_gps_log = [], [], []
z_alt_log, t_alt_log = [], []


# ------------------------- Functions ---------------------------------
def advance_waypoint(x, y, z):
    """Advance to next waypoint once within acceptance radius. Updates global target."""
    global wp_index, x_target, y_target, z_target
    if wp_index < len(waypoints_list) - 1:
        dist = np.sqrt((x - x_target)**2 + (y - y_target)**2 + (z - z_target)**2)
        if dist < wp_accept_radius:
            wp_index += 1
            x_target, y_target, z_target = waypoints_list[wp_index]

def rotation_matrix(phi, theta, psi): # 
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


def PID(setpoint, pv, kp, ki, kd, prev_error, integral, dt, derivative_prev, alpha,
        out_min=-np.inf, out_max=np.inf, prev_pv=None):
    error = setpoint - pv
    integral += error * dt
    # Derivative-on-measurement: use -Δpv/dt instead of Δerror/dt.
    # When the setpoint steps (new waypoint), pv doesn't move instantly so the
    # derivative is zero — no kick. Only real motion of the drone triggers D.
    if prev_pv is not None:
        derivative_raw = -(pv - prev_pv) / dt
    else:
        derivative_raw = (error - prev_error) / dt  # fallback on first step only
    derivative_filtered = alpha * derivative_prev + (1 - alpha) * derivative_raw
    control = kp * error + ki * integral + kd * derivative_filtered
    # Conditional anti-windup: undo integral accumulation when output is saturated.
    # Prevents the integrator from winding up during large-angle transients.
    if control > out_max or control < out_min:
        integral -= error * dt
    control = np.clip(control, out_min, out_max)
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

# Kalman Filter Helper Functions
def build_F(dt):
    F = np.eye(6)
    F[0:3, 3:6] = np.eye(3) * dt
    return F

def build_B(dt):
    B = np.zeros((6, 3))
    B[0:3, :] = 0.5 * dt**2 * np.eye(3)  # Position affected by accel
    B[3:6, :] = dt * np.eye(3)            # Velocity affected by accel
    return B

def build_Q(dt):
    sigma_acc = 0.1  # m/s², process noise std dev for acceleration
    Q_pos = 0.25 * dt**4 * sigma_acc**2
    Q_vel = dt**2 * sigma_acc**2
    Q = np.zeros((6, 6))
    Q[0:3, 0:3] = np.eye(3) * Q_pos
    Q[3:6, 3:6] = np.eye(3) * Q_vel
    Q[0:3, 3:6] = np.eye(3) * 0.5 * dt**3 * sigma_acc**2  # Cross-covariance
    Q[3:6, 0:3] = np.eye(3) * 0.5 * dt**3 * sigma_acc**2  # Cross-covariance
    return Q

def predict(x, P, a_world, dt):
    F = build_F(dt)
    B = build_B(dt)
    Q = build_Q(dt)
    x_pred = F @ x + B @ a_world # State prediction
    P_pred = F @ P @ F.T + Q # Covariance prediction
    return x_pred, P_pred

# --------------------- Main Controls ------------------------
gamma = k_torque / k_thrust
B_mixer = mixer_matrix(rotors, gamma, L)

# Main Control Loop - MATLAB Drone Simulation and Control Solution
for i in range(steps):
    time = i * dt
    advance_waypoint(x, y, z)
    # Sensor Updates
    # Altimeter Update
    alt_timer += dt
    if alt_timer >= 1.0 / alt_update_rate:
        alt_dt = 1.0 / alt_update_rate
        alt_timer = 0.0
        z_meas_new = z + np.random.normal(0.0, alt_noise_std)
        vz_meas = (z_meas_new - z_meas) / alt_dt
        z_meas = z_meas_new
        z_alt_log.append(z_meas_new)
        t_alt_log.append(time)
    # GPS Update
    gps_timer += dt
    if gps_timer >= 1.0 / gps_update_rate:
        gps_dt = 1.0 / gps_update_rate
        gps_timer = 0.0
        x_meas_new = x + np.random.normal(0.0, gps_noise_std)
        y_meas_new = y + np.random.normal(0.0, gps_noise_std)
        # Velocity from consecutive GPS positions divided by the GPS interval (0.2s),
        # NOT by dt (0.01s) — dividing a 0.2s displacement by 0.01s gives 20x velocity spikes.
        vx_meas = (x_meas_new - x_meas) / gps_dt
        vy_meas = (y_meas_new - y_meas) / gps_dt
        x_meas = x_meas_new
        y_meas = y_meas_new
        x_gps_log.append(x_meas_new)
        y_gps_log.append(y_meas_new)
        t_gps_log.append(time)
    # IMU Update — gyro and accelerometer run together at IMU rate
    gyro_timer += dt
    if gyro_timer >= 1.0 / gyro_update_rate:
        gyro_timer = 0.0

        # --- Gyro model ---
        # Bias drifts as a random walk (Brownian motion): σ_bias = drift * √dt
        gyro_bias_p += np.random.normal(0.0, gyro_bias_drift * np.sqrt(dt))
        gyro_bias_q += np.random.normal(0.0, gyro_bias_drift * np.sqrt(dt))
        gyro_bias_r += np.random.normal(0.0, gyro_bias_drift * np.sqrt(dt))
        p_meas = p_rate + gyro_bias_p + np.random.normal(0.0, gyro_noise_std)
        q_meas = q_rate + gyro_bias_q + np.random.normal(0.0, gyro_noise_std)
        r_meas = r_rate + gyro_bias_r + np.random.normal(0.0, gyro_noise_std)

        # --- Accelerometer model ---
        # Measures specific force in body frame = (linear_accel - gravity) in body frame.
        # NOTE: tilt-from-accel (phi_acc, theta_acc) is only valid when linear accel ≈ 0
        # (hover). During translation, linear accel corrupts the CF attitude estimate —
        # this is the physically accurate accelerometer behaviour.
        R_imu = rotation_matrix(phi, theta, psi)
        # Specific force = (linear_accel_world - gravity_world) rotated to body frame.
        # Uses ax/ay/az from the previous physics step (one-step delay, negligible at 100 Hz).
        specific_force_body = R_imu.T @ (np.array([ax, ay, az]) - np.array([0.0, 0.0, g]))
        ax_meas = specific_force_body[0] + np.random.normal(0.0, accel_noise_std)
        ay_meas = specific_force_body[1] + np.random.normal(0.0, accel_noise_std)
        az_meas = specific_force_body[2] + np.random.normal(0.0, accel_noise_std)

        # Complementary Filter for Attitude Estimation
        euler_rates_cf = body_to_euler_rates(phi_meas, theta_meas, p_meas, q_meas, r_meas)
        phi_gyro = phi_meas + euler_rates_cf[0] * dt  # integrate gyro rates to get angle change
        theta_gyro = theta_meas + euler_rates_cf[1] * dt
        # Subtract estimated linear acceleration (from previous KF step) so the
        # accelerometer only sees the gravity vector. Without this correction,
        # horizontal linear acceleration biases the tilt estimate, creating a
        # positive-feedback loop: accel → wrong tilt → wrong torque → more accel.
        a_body_est = R_imu.T @ a_world_prev          # body-frame linear accel estimate
        sf_corr = np.array([ax_meas, ay_meas, az_meas]) - a_body_est.flatten()
        phi_acc   = np.arctan2(sf_corr[1], sf_corr[2])
        theta_acc = np.arctan2(-sf_corr[0], np.sqrt(sf_corr[1]**2 + sf_corr[2]**2))
        phi_meas   = alpha_cf * phi_gyro   + (1 - alpha_cf) * phi_acc
        theta_meas = alpha_cf * theta_gyro + (1 - alpha_cf) * theta_acc
        psi_meas   = psi    # <-- magnetometer/GPS needed to correct yaw drift

        # Kalman Filter for Position Estimation
        a_world = R_imu @ np.array([[ax_meas], [ay_meas], [az_meas]]) + np.array([[0], [0], [g]])  # convert accel to world frame and remove gravity
        x_state, P = predict(x_state, P, a_world, dt)
        a_world_prev = a_world  # save for CF correction next step
        if gps_timer <= 0.0:
            z_gps = np.array([[x_meas_new], [y_meas_new]])  # 2D: x,y only
            H_gps = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]])  # Measures x, y only
            K_gps = P @ H_gps.T @ np.linalg.inv(H_gps @ P @ H_gps.T + R_gps)
            x_state = x_state + K_gps @ (z_gps - H_gps @ x_state)
            P = (np.eye(6) - K_gps @ H_gps) @ P
        if alt_timer <= 0.0:
            z_alt = np.array([[z_meas_new]])
            H_alt = np.array([[0, 0, 1, 0, 0, 0]])  # Measurement matrix for altitude
            R_alt = np.array([[alt_noise_std**2]])  # Measurement noise covariance for altitude
            K_alt = P @ H_alt.T @ np.linalg.inv(H_alt @ P @ H_alt.T + R_alt)
            x_state = x_state + K_alt @ (z_alt - H_alt @ x_state)
            P = (np.eye(6) - K_alt @ H_alt) @ P
        



    # Position Control (Outer Loop)
    # Use Kalman Filter state for position and velocity feedback.
    # KF runs at 100 Hz (every control step) and fuses accelerometer prediction
    # with 5 Hz GPS and 50 Hz altimeter corrections, giving smooth continuous
    # estimates. Raw GPS velocity (Δx / 0.2 s) has ~0.7 m/s noise std which
    # overwhelms the D term; KF velocity noise is ~0.005 m/s after 1 s.
    x_est  = x_state[0, 0];  vx_est = x_state[3, 0]
    y_est  = x_state[1, 0];  vy_est = x_state[4, 0]
    z_est  = x_state[2, 0];  vz_est = x_state[5, 0]

    # Z-axis Control
    thrust_feedback = np.clip(
        kp_z * (z_target - z_est) - kd_z * vz_est,
        -m * abs(g), 15.0 - m * abs(g))
    # Feedforward + Feedback with tilt compensation.
    # When tilted, vertical thrust = T·cos(φ)·cos(θ). Scale up so the vertical
    # component always equals m·|g| + feedback, regardless of attitude.
    # Capped at cos=0.5 (60° tilt) to prevent runaway at extreme angles.
    tilt_comp = 1.0 / max(np.cos(phi_meas) * np.cos(theta_meas), 0.5)
    total_thrust = np.clip((m * abs(g) + thrust_feedback) * tilt_comp, 0.0, 15.0)
    # X-axis Control -> Desired Pitch Angle
    theta_desired = np.clip(
        kp_x * (x_target - x_est) - kd_x * vx_est,
        np.radians(-25), np.radians(25))
    # Y-axis Control -> Desired Roll Angle
    phi_desired = np.clip(
        -(kp_y * (y_target - y_est) - kd_y * vy_est),
        np.radians(-25), np.radians(25))
    # # Yaw Setpoint
    psi_desired = 0.0  # psi_target

    # Attitude Control (Inner Loop)
    # Roll Control (phi) -> Roll moment
    tau_phi_cmd, prev_error_phi, integral_phi, derivative_prev_phi = PID(
        phi_desired, phi_meas, kp_phi, ki_phi, kd_phi, prev_error_phi, integral_phi, dt, derivative_prev_phi, alpha_att,
        prev_pv=prev_pv_phi)
    prev_pv_phi = phi_meas

    # Pitch Control (theta) -> Pitch moment
    tau_theta_cmd, prev_error_theta, integral_theta, derivative_prev_theta = PID(
        theta_desired, theta_meas, kp_theta, ki_theta, kd_theta, prev_error_theta, integral_theta, dt, derivative_prev_theta, alpha_att,
        prev_pv=prev_pv_theta)
    prev_pv_theta = theta_meas

    # Yaw Control (psi) -> Yaw moment
    tau_psi_cmd, prev_error_psi, integral_psi, derivative_prev_psi = PID(
        psi_desired, psi_meas, kp_psi, ki_psi, kd_psi, prev_error_psi, integral_psi, dt, derivative_prev_psi, alpha_att,
        prev_pv=prev_pv_psi)
    prev_pv_psi = psi_meas

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
    wp_index_log.append(wp_index)
    x_kf_log.append(x_state[0, 0])
    y_kf_log.append(x_state[1, 0])
    z_kf_log.append(x_state[2, 0])

# ----------------- Performance Metrics ---------------------
wp_index_arr = np.array(wp_index_log)
t_arr        = np.array(time_log)
pos_arr      = np.column_stack([x_log, y_log, z_log])

print("\n--- Waypoint Mission Summary ---")
for wi, wp in enumerate(waypoints_list):
    wp_pt = np.array(wp)
    indices = np.where(wp_index_arr == wi)[0]
    if len(indices) == 0:
        print(f"  WP{wi} {wp}: not reached")
        continue
    t_enter = t_arr[indices[0]]
    # Settling: first timestep (while at this WP) after which error stays <= accept radius
    err_wp = np.linalg.norm(pos_arr[indices] - wp_pt, axis=1)
    settled = None
    for j in range(len(err_wp)):
        if np.all(err_wp[j:] <= wp_accept_radius):
            settled = t_arr[indices[j]]
            break
    settled_str = f"t={settled:.1f}s" if settled is not None else "never"
    print(f"  WP{wi} {wp}: arrived t={t_enter:.1f}s | settled {settled_str}")

err_final = np.linalg.norm(pos_arr[-1] - np.array(waypoints_list[-1]))
print(f"\nFinal position error to last WP: {err_final:.3f} m")

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


# ---------------------- KF vs Truth vs Sensors ----------------------
fig_kf, axs_kf = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig_kf.suptitle('Kalman Filter: Estimated vs True Position vs Sensor Readings', fontsize=13)

axs_kf[0].plot(time_log, x_log,    'b-',  linewidth=1.5, label='True x')
axs_kf[0].plot(time_log, x_kf_log, 'r-',  linewidth=1.5, label='KF estimate x', alpha=0.85)
axs_kf[0].scatter(t_gps_log, x_gps_log, c='limegreen', s=18, zorder=5, label='GPS x', marker='x')
axs_kf[0].set_ylabel('X Position (m)')
axs_kf[0].legend(loc='upper right')
axs_kf[0].grid(True, alpha=0.4)

axs_kf[1].plot(time_log, y_log,    'b-',  linewidth=1.5, label='True y')
axs_kf[1].plot(time_log, y_kf_log, 'r-',  linewidth=1.5, label='KF estimate y', alpha=0.85)
axs_kf[1].scatter(t_gps_log, y_gps_log, c='limegreen', s=18, zorder=5, label='GPS y', marker='x')
axs_kf[1].set_ylabel('Y Position (m)')
axs_kf[1].legend(loc='upper right')
axs_kf[1].grid(True, alpha=0.4)

axs_kf[2].plot(time_log, z_log,    'b-',  linewidth=1.5, label='True z')
axs_kf[2].plot(time_log, z_kf_log, 'r-',  linewidth=1.5, label='KF estimate z', alpha=0.85)
axs_kf[2].scatter(t_alt_log, z_alt_log, c='orange', s=8, zorder=5, label='Altimeter z', alpha=0.6)
axs_kf[2].set_ylabel('Z Position (m)')
axs_kf[2].set_xlabel('Time (s)')
axs_kf[2].legend(loc='upper right')
axs_kf[2].grid(True, alpha=0.4)

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

# Pre-create persistent artist objects — avoids ax.clear() on every frame
arm_lines = [ax.plot([], [], [], 'k-', linewidth=3)[0] for _ in range(4)]
brace_line1, = ax.plot([], [], [], 'k-', linewidth=3, alpha=0.6)
brace_line2, = ax.plot([], [], [], 'k-', linewidth=3, alpha=0.6)
trail_line,  = ax.plot([], [], [], 'g-', alpha=0.5, linewidth=2)

# Static waypoint path — draw once, never updated
wp_arr = np.array(waypoints_list)
ax.plot(wp_arr[:, 0], wp_arr[:, 1], wp_arr[:, 2],
        'o--', color='gray', markersize=5, linewidth=1, alpha=0.5)

# Active waypoint indicator — updated each frame
active_wp_dot, = ax.plot([], [], [], 'o',
                          markeredgecolor='orange', markerfacecolor='none',
                          markeredgewidth=3, markersize=14)

# Static axis setup — done once, not repeated per frame
max_range = max(5.0, float(np.max(np.abs(wp_arr[:, :2]))) * 1.4,
                      float(np.max(wp_arr[:, 2])) * 1.4)
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([0, max_range])
ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_zlabel('Z (m)', fontsize=12)
ax.grid(True, alpha=0.3)
title = ax.set_title('', fontsize=14)

all_artists = arm_lines + [brace_line1, brace_line2, trail_line, active_wp_dot, title]


def update_frame(frame):
    idx = min(frame * 5, len(x_log) - 1)

    pos = np.array([x_log[idx], y_log[idx], z_log[idx]])
    R = rotation_matrix(phi_log[idx], theta_log[idx], psi_log[idx])
    motor_pos_world = (R @ motor_pos_body.T).T + pos

    # Update arm lines (center → each motor)
    for i, line in enumerate(arm_lines):
        line.set_data_3d([pos[0], motor_pos_world[i, 0]],
                         [pos[1], motor_pos_world[i, 1]],
                         [pos[2], motor_pos_world[i, 2]])

    # Update cross-bracing
    brace_line1.set_data_3d([motor_pos_world[0, 0], motor_pos_world[1, 0]],
                             [motor_pos_world[0, 1], motor_pos_world[1, 1]],
                             [motor_pos_world[0, 2], motor_pos_world[1, 2]])
    brace_line2.set_data_3d([motor_pos_world[2, 0], motor_pos_world[3, 0]],
                             [motor_pos_world[2, 1], motor_pos_world[3, 1]],
                             [motor_pos_world[2, 2], motor_pos_world[3, 2]])

    # Update trajectory trail
    trail_start = max(0, idx - 200)
    trail_line.set_data_3d(x_log[trail_start:idx],
                           y_log[trail_start:idx],
                           z_log[trail_start:idx])

    # Update active waypoint marker
    curr_wp = waypoints_list[wp_index_log[idx]]
    active_wp_dot.set_data_3d([curr_wp[0]], [curr_wp[1]], [curr_wp[2]])

    title.set_text(f'Time: {time_log[idx]:.2f}s | WP{wp_index_log[idx]} | Alt: {z_log[idx]:.2f}m | '
                   f'Phi: {np.degrees(phi_log[idx]):.1f}° | Theta: {np.degrees(theta_log[idx]):.1f}°')

    return all_artists


# Create animation
num_frames = len(x_log) // 5
anim = animation.FuncAnimation(fig, update_frame, frames=num_frames,
                               interval=50, blit=False)

plt.show()