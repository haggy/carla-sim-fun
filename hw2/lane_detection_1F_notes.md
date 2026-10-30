# Lane detection: parameter tuning and failure cases (unit 1F)

Figures referenced below are in `report_figures/1F/` (PNGs are git-ignored, so they only exist locally).

## How the parameters were evaluated

Frames were captured from the hw2 test-agent camera (320x240, FOV 90, 2 m high) on autopilot drives through
Town01 (the benchmark town) and Town10HD. For every frame the ground-truth left and right edges of the ego lane
were taken from the CARLA map (`waypoint.next()` +- half the lane width) and projected into the same cut/BEV pixel
coordinates the detector works in.

- Frames inside junctions (no single ego lane) and frames standing still (duplicates) are excluded.
- Scored set: 94 Town01 and 78 Town10 frames, two boundaries each.
- A boundary counts as **correct** if its spline lies within 0.25 m (mean distance from the ground-truth points) over
  more than half of the ground-truth rows in the 4.5-13.5 m detection range.

## Gradient threshold

Success rate per town (spline smoothness 50). Columns: without the lateral gate added in 1F, and with the final
gate (`max_lateral_jump_px=4`, `max_lateral_slope=0.3`).

| threshold | Town01 (no gate) | Town10 (no gate) | Town01 (gate) | Town10 (gate) | all (gate) |
|---:|---:|---:|---:|---:|---:|
| 6  | 40.4% | 14.1% | 47.3% | 21.2% | 35.5% |
| 8  | 47.9% | 32.7% | 54.3% | 37.2% | 46.5% |
| **10** | 47.3% | 46.8% | **53.2%** | **54.5%** | **53.8%** |
| 12 | 38.3% | 54.5% | 45.2% | 57.7% | 50.9% |
| 14 (template) | 28.7% | 59.0% | 33.0% | 60.9% | 45.6% |
| 20 | 6.4% | 57.7% | - | - | - |
| 30 | 2.7% | 41.7% | - | - | - |

- The towns disagree. Town01 (wet, overcast, faded markings) needs a low threshold, because its markings barely
  clear 14. Town10's high-contrast markings do best at 12-14, and lower thresholds admit shadow and texture edges.
- **Chosen: 10.** It is the best overall and close to each town's optimum. The template default of 14 scores only
  28.7% on Town01, which is the town the benchmark runs on.

## Spline smoothness

Accuracy was the same for every `s` from 1 to 30000 (53.2% / 54.5%). Smoothness only controls how much the spline
follows the 1 px quantisation of the maxima. Measured over the 295 boundaries with at least 30 points (median 71):

| s | interior knots | total heading change over the boundary |
|---:|---:|---:|
| 1 | 25.7 | 403 deg |
| 10 (template) | 4.5 | 52 deg |
| 30 | 1.1 | 22 deg |
| **100** | 0.3 | 18 deg |
| 1000 | 0.0 | 18 deg |

- From about 100 upwards the spline stops tracking pixel noise. The remaining ~18 deg is the real road curvature.
- This matches scipy's guidance of `s` approximately equal to the number of points for unit-variance noise.
- **Chosen: 100.** Jittery splines would feed straight into the waypoint and steering stages.

## Result

| parameters | Town01 | Town10 | all |
|---|---:|---:|---:|
| template defaults (threshold 14, s 10, no gate) | 28.7% | 58.3% | 42.2% |
| final (threshold 10, s 100, gate 4 px + 0.3/row) | 53.2% | 54.5% | 53.8% |

Median error of the correct boundaries: 0.07 m (Town01) and 0.16 m (Town10).

## Failure cases and reasons

Figures: `gt_comparison_town01.png` and `gt_comparison_town10.png` show, per frame, the front view and the BEV
detection range with ground truth (green = left, magenta = right), assigned maxima (cyan/yellow), all maxima (red),
fitted splines (orange) and projected obstacles (purple). `live_montage.png` and `live/` are snapshots of the live
test window.

1. **Cast shadows** (Town10 g1032, g1036, g1140, g1159). Tree and building shadows have long, sharp, nearly straight
   edges that are much stronger than the paint. The gradient detector cannot tell them apart from markings, and a
   shadow edge near the car is often taken as the start point.
2. **Faint or wet markings and puddles** (Town01 g0048, g0079, g0101, g0111). Town01's yellow dashes and the right
   edge give gradients close to the threshold. Puddle outlines and rain texture are just as strong, so any threshold
   low enough for the paint also admits clutter. With the lateral gate the boundary no longer zigzags through the
   clutter, but a start on a puddle edge still gives a wrong, short boundary.
3. **Dashed lines with long gaps / boundary not visible next to the car.** The first rows only cover +-4.5 m
   laterally, so a dashed line in a gap is invisible there. The start-point search then picks the next visible edge,
   often the yellow centre line one lane over or the curb (Town10 g1004, g1100). Retrying from further start
   candidates fixes some of these, but not all.
4. **Crosswalks, stop lines, intersection grids and arrows** (`live/step_00200`-`00440`, earlier 1D sheets f005,
   f060, f090). Dozens of parallel stripes produce maxima on every row. Nearest-neighbour assignment hops between
   stripes and the spline wobbles. Junction frames are excluded from the scores above, but this is the most visible
   failure while driving through town.
5. **Occlusion by vehicles.** Obstacle suppression (1C) removes the car's outline but also the lane paint behind it,
   so a boundary hidden by a lead vehicle falls back to the preceding frame's spline (drawn dashed in the test
   window).
6. **Flat-ground assumption.** The BEV homography assumes a level road and a level camera. Pitching under braking or
   acceleration and road grade bend or splay parallel lines in the BEV, which adds curvature and widens the gap
   between the boundaries.
7. **Ego hood detected as a car.** In about 10% of frames YOLO labelled the ego hood a car. Its mask blanked out the
   edges right in front of the vehicle. This was fixed in 1F by dropping detections that touch the bottom image
   edge but start in its lowest quarter.

## Reproducing

```bash
cd "hw2/02_modular_code _template"
python test_lane_detection.py --seed 11                        # autopilot
python test_lane_detection.py --manual                         # arrow keys (focus the plot window); 's' saves a snapshot
python test_lane_detection.py --snapshot-every 40 --snapshot-dir ../report_figures/1F/live
```
