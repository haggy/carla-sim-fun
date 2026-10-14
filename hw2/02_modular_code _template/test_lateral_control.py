from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
from lateral_control import LateralController
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key

# action variables

# init carla environement

# define variables
steps = 0

# init modules of the pipeline
LD_module = LaneDetection()
LatC_module = LateralController()

# init extra plot
fig = plt.figure()
plt.ion()
plt.show()

while True:
    # perform step

    # lane detection
    lane1, lane2 = LD_module.lane_detection(s)

    # waypoint and target_speed prediction
    waypoints = waypoint_prediction(lane1, lane2)
    target_speed = target_speed_prediction(waypoints)

    # control with constant gas and no braking
    a[0] = LatC_module.stanley(waypoints, speed)

    # output and plot figure
    if steps % 2 == 0:
        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        print("targetspeed {:+0.2f}".format(target_speed))
        LD_module.plot_state_lane(s, steps, fig, waypoints=waypoints)
    steps += 1

    # check if stop