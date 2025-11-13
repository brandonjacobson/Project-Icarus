# Vertical Hover PID Notes
This test shows off a quadcopter hover at 10 meters using a PID controller. The dynamics are based off the simple vertical dynamics solved in Icarus-Dynamics.pdf.

For this test, a PID controller was created where the error is the altitude minus the desired altitude. After writing the standard PID control algorithm, I had to tune
the gain to eliminate the large oscillations and get close to our target altitude. At first, I increased the kp value to 2.0 while keeping the others at 0.0, which proportionally multiplies the error, returning 
this as the thrust value. This led the simulated quad to fly up towards the target and then begin oscillating greatly. I then increased the kd value to 2.0, which multiplies the 
time derivative of the error, meaning that the closer we get to the target, the less we should correct, eliminating the large oscillations. Suprisingly, these first values led to
a very quick response and minimal oscillation.

To add fidelity and more precisely tune the PID, I implemented performance metrics, including overshoot percentage and steady-state error. To make it a more real scenario as well,
I implemented wind disturbance as a constant force of 1 Newton, occuring 20 seconds into the simulation. Since I had not used a ki term, the PID could not account for this 
and resulted in a steady state error of about 0.5 meter. Then, based on PID literature guidance, I added about 0.05 ki gain, which brought the steady state error down to 0.16 meter.

The next step was to make the PID even more robust to disturbances. I made the ki 0.06 to inch closer towards 0 steady-state but this now added overshoot on the undisturbed
part of flight. With these gains, I had an overshoot of about 2.8%. The next step was to add integral anti-windup. This works as the integral term slowly builds up over time
to account for the previous error, but over a long period of time, this integral can get very large, especially with a steady-state error. We use clamping on this term, limiting it
to between +- 30. This, combined with a slight increase in kp and kd gains results in an overshoot of only 1.7% and steady-state error of 0.1 meter. 

Finally the last step in increasing fidelity of the simulation is sensor noise. This is modelled using a 50 Hz frequency update cycle for our "altimeter", adding a random value between
0 and 0.05 meters to the altitude value every 1/50 seconds, similar to the noise you might see in a real altimeter. After implementing this, the PID was completely off and
the noise needed filtering to allow our controls to function. To do this, I implemented a low-pass filter on the derivative value. This looks like a gain value, alpha (between 0 and 1),
multiplied by the previous iteration's derivative value plus (1-alpha) multiplied by the current calculated derivative. A higher gain value relies more on the previous derivative,
smoothing out the controls, not taking current jumps in slope to fully alter the control. Since our noise is very volatile, an alpha of 0.9 results in a clean, usable altitude model.

Final Values:
kp = 2.2
ki = 0.06
kd = 2.7
clamp = +- 30
alpha = 0.9

Key Insights:
1. Derivative term is critical for damping oscillations
2. Integral term necessary for disturbance rejection but causes overshoot
3. Anti-windup prevents integral term from growing unbounded
4. Raw sensor data is too noisy for derivative calculation - filtering essential
5. Tradeoff between responsiveness and stability
