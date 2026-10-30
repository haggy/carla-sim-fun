# Lateral control notes (units 3B, 3C)

All tests ran on Town01. Figures are in `report_figures/3B/` and `report_figures/3C/` (git-ignored PNGs).

## 3B: Stanley controller

`LateralController.stanley(waypoints, speed)` implements eq. 3, delta = psi + arctan(k d / v):

- **Orientation error psi:** the angle of the first path segment (waypoint 1 minus waypoint 0) relative to the
  car's +y axis in the BEV. The car always faces +y in the state image, so its own heading is 0.
- **Cross-track error d:** the lateral offset of waypoint 0 (spline parameter 0) from the car position (160, 0).
  It is converted to metres (0.075 m per BEV pixel) so that k d / v is dimensionless with v in m/s, which makes k
  a rate in 1/s.
  - Waypoint 0 is the first BEV row, 4.5 m ahead of the vehicle origin, so d is measured at a look-ahead point
    rather than at the front axle.
- **Sign:** both terms are positive when the path lies or heads to the right. That matches CARLA's steer sign.
- **Output:** the wheel angle is clipped to +-0.4 rad (as in the template) and divided by the MKZ's 70 deg full
  lock. The command is therefore in CARLA's [-1, 1] at the real wheel angle, with a limit of +-0.33. The template's
  `/ 0.4` would have mapped 0.4 rad onto the full 70 deg, silently amplifying every command about 3x.
- **Test:** `test_lateral_control.py` steers with Stanley at constant gas and no braking.
  - Gas is 0.6, released above 30 km/h, which is the Town01 limit. At the template agent's 0.4 the MKZ needs about
    10 s to reach 10 km/h. At that speed the k d / v term saturates on any error.
  - There are no NPCs, because the car does not brake.
  - `--path ground-truth` replaces the detected waypoints with the lane centre from the CARLA map, over the same
    4.5-13.5 m range and in the same coordinates.
  - At the end the test prints lane-keeping statistics against the map: share of steps in lane (the car body
    inside its own lane, oncoming lane counted as out), distance driven before first leaving the lane, mean
    |offset| from the lane centre while in lane, and steering jitter (mean |change of steer| per step).
  - Junction steps are excluded, because the map's lane centre there depends on the chosen branch.

### Gain, tuned on the ground-truth path

With perfect waypoints the controller is judged on its own. Three spawn points (seeds 1-3), 600 steps (30 s) each;
means over the seeds:

| k | in lane | distance before leaving lane | mean abs. offset | steering jitter |
|---:|---:|---:|---:|---:|
| 0.3 | 96.4% | 109 m | 0.22 m | 0.004 |
| 1 | 95.3% | 131 m | 0.15 m | 0.006 |
| **3** | 95.7% | 131 m | 0.12 m | 0.007 |
| 10 | 97.9% | 110 m | 0.11 m | 0.011 |
| 30 | 98.4% | 147 m | 0.10 m | 0.103 |

- **Small gains** leave a steady offset after every disturbance, which is slowly corrected (0.3: 0.22 m).
- **Large gains** hold the centre more tightly, but the steering starts to oscillate. k = 30 jitters 15x as much as
  k = 3, because a few centimetres of offset already produce a large arctan term at 8 m/s.
- **The departures are the same at every gain.** They happen at Town01's 90 deg junction turns (`gt_k3_s2.png`).
  At 30 km/h the needed wheel angle exceeds the 0.4 rad clip, so the car runs up to 2.4 m wide out of the corner
  and then recovers. That is a speed problem: the longitudinal controller (2.4) has to slow down for the turn.
- **Chosen: k = 3.** It is close to the best tracking without the oscillation of large gains. It also limits the
  amplification of perception noise (below): with d noise of ~0.4 m at 8 m/s, k = 3 gives arctan(0.15) = 0.15 rad,
  while k = 10 already saturates.

### Behaviour on the detected path

Same setup, steering on the perception pipeline's smoothed waypoints:

| k | in lane | distance before leaving lane | total distance in 30 s |
|---:|---:|---:|---:|
| 0.5 | 56.5% | 11 m | 37 m |
| 1 | 33.8% | 11 m | 35 m |
| 2 | 31.1% | 11 m | 37 m |
| 3 | 47.1% | 11 m | 38 m |
| 4 | 36.2% | 6 m | 38 m |

With these perception errors the gain hardly matters: every run leaves the lane within about 20 m. Behaviour
(`montage_s2.png`, `detected_k3_s*.png`):

- **The steering switches between full left and full right lock every few frames.** The car weaves +-1 m across the
  lane at about 1 Hz, and usually ends against a curb or pole on the right or in the oncoming lane.
- **At launch** the speed is near zero, so the arctan term saturates for any non-zero d. The first second is
  driven at full lock, in whichever direction the first detection points.

The cause is the detected path, not the controller:

- **The controller is correct.** On the ground-truth path the same code drives the full 223 m route.
- **Shadow test, perception path.** The autopilot drove Town01 (seeds 1 and 3, 471 non-junction frames) while
  psi and d were computed from the detected path without steering. The autopilot held the car within 0.17 m (std)
  of the lane centre and 0.9 deg of its heading, yet the detected path reported:
  - a median |d| of 0.43 m and a median |psi| of 0.13 rad (7.5 deg);
  - a std of 0.77 m and 0.34 rad;
  - frame-to-frame jumps of 0.44 m and 0.25 rad;
  - essentially no correlation with the true offset or heading (|r| < 0.15).

  The errors are the lane detection failures from 1F: only ~55% of boundaries are correct. When a boundary is
  wrong, the centre waypoints jump laterally, and the first segment turns by tens of degrees.
- **Why the controller makes it worse.** Stanley uses the error at the first waypoint and the first segment
  directly, without filtering, so every jump becomes a steering step. The 0.4 rad clip turns large jumps into
  full-lock commands. Because lane detection is relative to the car, a car that has drifted towards the curb then
  finds the curb and sidewalk edges as "its" lane. The error therefore reinforces itself until the car leaves the
  road.
- Damping (3C) attacks the frame-to-frame jumps. Better perception, or temporal filtering of the boundaries, is
  needed for the steady drift.
