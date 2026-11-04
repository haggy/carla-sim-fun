# Path planning notes (units 2A, 2B, 2C)

Figures are in `report_figures/2A/`, `report_figures/2B/` and `report_figures/2C/` (git-ignored PNGs). All tests ran on Town01.

## 2A: road centre waypoints

`waypoint_prediction(..., way_type="center")` samples both lane boundary splines at 6 equidistant spline parameters
`t = 0, 0.2, ..., 1` (so `t = 0` is included) and returns the midpoints of points with the same parameter. The
result is a (2 x 6) array in the lane detector's cut coordinates: x in BEV pixels, y in rows ahead of the car, with
the car at (160, 0) heading +y.

Run: `python test_waypoint_prediction.py --way-types center` (white waypoints, orange boundaries).

### When it fails, and why (`center_waypoints_montage.png`)

- **Boundaries of different length** (step 75).
  - `splprep` normalises each spline's parameter to that boundary's own length. `t = 0.4` is therefore 40% of the
    way along the left boundary but also 40% along the right one, even if one is 9 m long and the other 2 m.
  - Pairing points with the same `t` then pairs points at different distances ahead. The centre is pulled toward
    the car and sideways.
  - This happens whenever one boundary is dashed, partly occluded or only briefly detected.
- **Boundaries that start at different rows.** The same mismatch occurs when one boundary starts further ahead than
  the other.
- **Intersections and corners** (step 150). One boundary follows the curb around the corner while the other goes
  straight, so the "centre" cuts diagonally across the junction.
- **Wrong lane boundaries** (step 225). The waypoints can only be as good as the detected boundaries. Glare, wet
  reflections, shadows and crosswalks (see `lane_detection_1F_notes.md`) give crossing or misplaced boundaries,
  and the centre between them is meaningless.
- **One boundary lost.** If one boundary can't be fitted, the preceding frame's spline is reused. While driving
  that spline is stale, so the centre drifts.

## 2B: path smoothing

Objective (assignment eq. 1):

    argmin_{x_1..x_N}  sum_i |y_i - x_i|^2  -  beta * sum_n  (x_{n+1} - x_n) . (x_n - x_{n-1}) / (|x_{n+1} - x_n| |x_n - x_{n-1}|)

### Purpose of the second term

- Each summand is the normalised dot product of two consecutive path segments, i.e. the cosine of the angle the
  path turns at waypoint `x_n`: 1 when it goes straight on, 0 at a right angle, -1 when it doubles back.
- The sum is largest (N - 2) for a straight path. Subtracting it with weight beta therefore rewards paths that
  turn as little as possible.
- It is independent of segment length, so it penalises changes in direction only, not stretching or shrinking of
  the path.
- The first term pulls the waypoints toward the road centre `y_i`. The balance between the two straightens the path
  through bends: the path cuts the inside of a corner (the "racing line"). A straighter path has a smaller
  curvature, which permits higher speed (2.2c) and asks less of the steering.
- `curvature()` implements this second term. `waypoint_prediction(..., way_type="smooth")` minimises the objective
  with `scipy.optimize.minimize` (BFGS), starting from the centre waypoints.

### Implementation notes

- **Template scaling:** the template's `smoothing_objective` uses the mean of the squared distances rather than
  their sum. With 2 x 6 coordinates, its `weight_curvature = 40` equals beta = 480 in eq. 1.
- **Unit checks:** `curvature()` is 4.0 (= N - 2) for a straight line regardless of spacing, 3.0 with one 90 deg
  corner and 2.0 with a reversal. A straight centre path is returned unchanged.
- **Runtime:** about 9 ms per call.

### Effect of the curvature weight

Synthetic curved lane, 3.6 m wide, curving ~46 deg over the 9 m detection range:

| weight_curvature | total turning of the path | max. shift from centre | shift of the first waypoint |
|---:|---:|---:|---:|
| 0 | 45.7 deg | 0.00 m | 0.00 m |
| 10 | 43.3 deg | 0.05 m | 0.04 m |
| 40 (template) | 39.0 deg | 0.13 m | 0.13 m |
| 160 | 30.5 deg | 0.31 m | 0.31 m |
| 640 | 16.9 deg | 0.60 m | 0.60 m |
| 2560 | 6.0 deg | 0.81 m | 0.81 m |

- With the template weight the corner-cutting is mild: the path stays within ~0.13 m of the centre on the short
  9 m horizon.
- On the live drive (`center_vs_smooth_montage.png`) the visible effect is that kinks in the centre path are ironed
  out (step 50), while straight roads are unchanged (steps 0, 100).
- **The objective also moves the first waypoint.** Nothing pins it to the car's current position. Since the lateral
  controller (2.3) measures its cross-track error at that waypoint, a large weight would also shift the car's
  reference position. Pinning `x_1` (optimising only `x_2..x_N`) would avoid this if it causes problems in 2.3.

## 2C: target speed

`target_speed_prediction(waypoints)` implements eq. 2 with the assignment's initial parameters (v_max = 60,
v_min = 30, K_v = 4.5) on the first N = 5 waypoints:

    v_target = (v_max - v_min) * exp(-K_v * |N - 2 - curvature(x_1..x_N)|) + v_min

- `N - 2 - curvature` is 0 for a straight path. Each turn of angle theta adds `1 - cos(theta)`, about theta^2 / 2,
  so the speed falls with the square of the turning angles.
- The speeds are taken as km/h. CARLA reports m/s, so the longitudinal controller (2.4) has to convert.
- Run: `python test_waypoint_prediction.py --way-types center smooth`. The legend shows the target speed per
  waypoint type, and the title shows the current speed.

### Response on synthetic paths

The five waypoints, about 20 px (1.5 m) apart, turn steadily by the given total angle:

| total turning | target speed |
|---:|---:|
| 0 deg | 60.0 km/h |
| 20 deg | 59.0 km/h |
| 45 deg | 55.4 km/h |
| 90 deg | 45.5 km/h |

The squared angle makes the measure forgiving: a 90 deg bend spread over 3 turns of ~18 deg costs only
3 * (1 - cos 18 deg) = 0.15, so the speed drops by half its range.

### Live evaluation (Town01, autopilot, `report_figures/2C/`)

Three autopilot drives on Town01 (seeds 1-3, 800 steps each). There were 1443 moving frames with both boundaries
detected. For each frame the ground-truth heading change of the lane over the detection range (4.5-13.5 m ahead)
was read from the CARLA map:

| road ahead (ground truth) | frames | center: median / 10th pct | smooth: median / 10th pct |
|---|---:|---:|---:|
| straight (< 5 deg), outside junctions | 981 | 58.8 / 49.8 km/h | 59.6 / 56.5 km/h |
| straight, in a junction | 159 | 57.8 / 50.0 km/h | 59.3 / 57.0 km/h |
| bend, 5-45 deg | 166 | 57.7 / 47.7 km/h | 59.5 / 55.8 km/h |
| turn, >= 45 deg | 137 | 57.6 / 47.3 km/h | 59.4 / 56.2 km/h |

- **The predicted target speed hardly reacts to the real road geometry.**
  - The probability that a straight-road frame gets a higher target speed than a turn frame (ROC AUC) is only
    0.62 for the centre path and 0.59 for the smoothed path; 0.5 would be chance.
  - Changing K_v cannot fix this. It only rescales the exponent, so it doesn't change which frames get the lower
    speeds.
- **Why:**
  1. Town01's turns are 90 deg corners inside junctions. There, lane detection keeps following the straight curb
     or loses the boundary, so the detected path stays straight (step 575 of the montage: T-junction ahead,
     58-60 km/h). The path only bends once the car is already in the turn.
  2. The 9 m horizon covers only part of a bend. With the squared-angle measure above, that part costs little.
  3. On the centre path, detection noise (kinks between the waypoints) lowers the speed on straight roads about
     as often as real bends do (10% of straight frames are below 50 km/h). Smoothing (2B) removes the noise but
     also most of the real curvature.
- The defaults are therefore kept. Tuning K_v on this signal would only trade a lower straight-line speed against
  slightly earlier braking. A useful turn signal needs a longer or junction-aware horizon. The longitudinal
  controller should also cap the speed independently (Town01's limit is 30 km/h).
