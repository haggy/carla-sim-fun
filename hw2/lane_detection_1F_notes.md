# Lane detection: parameter tuning and failure cases (unit 1F)

All tuning and testing used **Town01 only**, as the assignment requires. Figures referenced below are in
`report_figures/1F/` (PNGs are git-ignored, so they only exist locally).

## How the parameters were evaluated

Frames were captured from the hw2 test-agent camera (320x240, FOV 90, 2 m high) on six autopilot drives through
Town01. For every frame the ground-truth left and right edges of the ego lane were taken from the CARLA map
(`waypoint.next()` +- half the lane width) and projected into the same cut/BEV pixel coordinates the detector
works in.

- Frames inside junctions (no single ego lane) and frames standing still (duplicates) are excluded.
- Scored set: 94 frames, i.e. 188 lane boundaries.
- A boundary counts as **correct** if its spline lies within 0.25 m (mean distance from the ground-truth points) over
  more than half of the ground-truth rows in the 4.5-13.5 m detection range.
- With 188 boundaries, one percentage point is about two boundaries, so differences of 1-2 points are within noise.

## Gradient threshold and lateral gate

Tracing a boundary through nearest-neighbour maxima zigzagged through puddle and rain-texture edges. 1F added a hard
gate: a candidate is rejected if its sideways offset from the boundary's extrapolated direction exceeds
`max_lateral_jump_px + max_lateral_slope * distance ahead`. Success rate (spline smoothness 100):

| threshold | no gate | gate 3 px + 0.2/row | gate 4 px + 0.3/row | gate 6 px + 0.5/row |
|---:|---:|---:|---:|---:|
| 4  | 24.5% | 28.7% | 28.7% | 27.1% |
| 6  | 39.9% | 50.0% | 47.3% | 45.7% |
| 7  | 47.3% | 54.3% | 50.0% | 51.1% |
| **8**  | 47.9% | **55.3%** | 54.3% | 51.1% |
| 9  | 46.8% | 54.8% | 52.1% | 50.5% |
| 10 | 46.3% | 54.3% | 53.2% | 50.0% |
| 12 | 38.3% | 45.7% | 45.2% | 42.0% |
| 14 (template) | 28.7% | 34.6% | 33.0% | 32.4% |

- **Threshold:** Town01 is wet and overcast with faded markings, and its yellow dashes and road edge barely clear the
  template threshold of 14. Below ~7, puddle outlines and rain texture swamp the markings.
- **Gate:** the gate adds 4-10 points at every threshold. The tightest gate tested is best and gives a flat plateau
  over thresholds 7-10.
- **Chosen:** gradient threshold **8** and gate **3 px + 0.2 px per row**.

## Spline smoothness

Accuracy did not depend on `s` (55.3-55.9% for `s` from 1 to 1000, within noise). Smoothness only controls how much
the spline follows the 1 px quantisation of the maxima. Measured over the 164 Town01 boundaries with at least
30 points (median 73):

| s | interior knots | total heading change over the boundary |
|---:|---:|---:|
| 1 | 16.9 | 264 deg |
| 10 (template) | 3.3 | 42 deg |
| 30 | 0.9 | 18 deg |
| **100** | 0.2 | 16 deg |
| 1000 | 0.0 | 16 deg |

- From about 100 upwards the spline stops tracking pixel noise. The remaining ~16 deg is the real road curvature.
- This matches scipy's guidance of `s` approximately equal to the number of points for unit-variance noise.
- **Chosen: 100.** Jittery splines would feed straight into the waypoint and steering stages.

## Result (Town01)

| parameters | correct boundaries | median error of correct ones |
|---|---:|---:|
| template defaults (threshold 14, s 10, no gate) | 28.7% | 0.12 m |
| final (threshold 8, s 100, gate 3 px + 0.2/row) | 55.3% | 0.07 m |

## Failure cases and reasons

Figures:
- `gt_comparison_town01.png`: per frame, the front view and the BEV detection range with ground truth
  (green = left, magenta = right), assigned maxima (cyan/yellow), all maxima (red), fitted splines (orange) and
  projected obstacles (purple).
- `live_montage.png` and `live/`: snapshots of the live test window.

1. **Puddles and rain texture** (g0018, g0048, g0101, g0111). Town01's wet road has puddle outlines and streaks as
   strong as the faded paint, so any threshold low enough for the paint also admits clutter. The lateral gate stops
   boundaries from zigzagging through it, but a start point on a puddle edge still gives a short, wrong boundary
   that the spline then extrapolates.
2. **Sparse, faded centre-line dashes** (g0043, g0064, g0079). The left boundary is the yellow centre line, made of
   short dashes with long gaps. When no dash lies in the first rows, the start-point search picks the nearest other
   edge. When the gap ahead is longer than the search range, the boundary stops after one dash and the spline only
   covers a few metres.
3. **Several parallel edges at the road side** (g0043, g0091). The right boundary is the curb. The curb face,
   gutter and sidewalk edge give 2-3 parallel edges a few pixels apart, so the traced boundary can sit on the
   sidewalk edge rather than the lane edge.
4. **Lead vehicles** (g0058). Obstacle suppression (1C) removes the car's outline but also the lane paint behind it,
   so a boundary hidden by a lead vehicle falls back to the preceding frame's spline (drawn dashed in the test
   window).
5. **Junctions** (`live/step_00150`, `step_00250`). The curb curves away into the side road, so one boundary follows
   the corner while the other continues straight.
6. **Flat-ground assumption.** The BEV homography assumes a level road and a level camera. Pitching under braking or
   acceleration bends or splays parallel lines in the BEV, which adds curvature and widens the gap between the
   boundaries.
7. **Ego hood detected as a car.** In about 10% of frames YOLO labelled the ego hood a car. Its mask blanked out the
   edges right in front of the vehicle. This was fixed in 1F by dropping detections that touch the bottom image
   edge but start in its lowest quarter.

## Reproducing

```bash
cd "hw2/02_modular_code _template"
python test_lane_detection.py --seed 4                         # autopilot, loads Town01
python test_lane_detection.py --manual                         # arrow keys (focus the plot window); 's' saves a snapshot
python test_lane_detection.py --snapshot-every 50 --snapshot-dir ../report_figures/1F/live
```
