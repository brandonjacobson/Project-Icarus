# Vertical Thrust Test Notes
This test shows off the motion of the quadcopter based on the simplified dyanmics found in dynamics.pdf. 

For this initial test, I show the motion if the thrust,or power input is toggled from Full Thrust (Calculated as 1.5 * m * g) to Hover (m*g) when the drone reaches
10 meters. The concept was to start simulating the first autonomous motion of the drone. I overlooked the concept however that the thrust input controls thrust *force*
instead of velocity. This meant that switching to Hover thrust would cancel out gravity, essentially keeping the drone moving upwards with its velocity until the thrust is
cut off after 5 seconds. This is evident in the graph found at Vertical_Thrust_Test1_Plot.png. 

To solve the drifting observed here there are a few evident options.
1. The Thrust input is cut off just before reaching the target height.
2. The Thrust is counteracted with a negative pulse of thrust right at the target height, similar to Bang-Bang controls.
3. The Thrust is slowly decreased as the quadcopter reaches the target height.

Approach 3 will be the most accurate and can be achieved using a PID controller.
