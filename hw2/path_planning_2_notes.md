# Path planning notes (units 2A, 2B)

Figures are in `report_figures/2A/` and `report_figures/2B/` (git-ignored PNGs). All tests ran on Town01.

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
