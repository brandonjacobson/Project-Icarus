# Project Icarus

A quadcopter flight dynamics simulator built from scratch in Python, focused on 
controls, state estimation, and autonomy research.

![Drone Trajectory Animation](simulations/drone_trajectory.gif)

## Overview

The simulator implements a full 6-DOF rigid body model of a quadrotor with:

- **Cascaded PID control** — outer position loop feeding desired attitude to an 
  inner attitude loop
- **Complementary filter** for attitude estimation, fusing gyroscope integration 
  with accelerometer tilt correction and linear acceleration compensation
- **Kalman filter** for position estimation, fusing 5 Hz GPS, 50 Hz altimeter, 
  and 100 Hz accelerometer predictions
- **Realistic sensor models** — gyro bias drift (random walk), accelerometer 
  specific force, GPS and altimeter noise
- **Waypoint mission system** with acceptance-radius-based sequencing
- **Motor dynamics** modelled as first-order lag with RPM saturation
- **Control allocation** via pseudoinverse mixer matrix

PID gains are derived analytically using pole placement on the linearised plant. 
The attitude inner loop targets ωn = 8 rad/s (settling ~0.7 s); the position outer 
loop targets ωn = 1.2 rad/s (settling ~3.7 s), satisfying the 5-10x cascade 
bandwidth separation rule.

## Roadmap

- [x] Cascaded PID control
- [x] Complementary filter attitude estimator
- [x] Kalman filter position estimator  
- [x] Waypoint mission sequencing
- [ ] LQR attitude and full-state control
- [ ] Minimum snap trajectory planning
- [ ] Extended Kalman Filter (EKF)

## Structure
Project-Icarus/
├── simulations/        # Flight dynamics simulator
├── fsw/                # Flight software
├── data_analysis/      # Post-processing and analysis tools
└── requirements.txt
## Getting Started

```bash
git clone https://github.com/brandonjacobson/Project-Icarus.git
cd Project-Icarus
pip install -r requirements.txt
python simulations/main_sim.py
```

## Dependencies

- numpy
- matplotlib
- scipy
- Pillow
