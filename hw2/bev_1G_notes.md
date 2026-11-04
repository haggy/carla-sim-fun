# Scene BEV from the front camera (unit 1G)

Code: `02_modular_code _template/bev.py` (`SceneBEV`). Figures are in `report_figures/1G/` (git-ignored PNGs).

## Method

- **Ground:** inverse perspective mapping with the flat-ground homography from 1A (`front2bev`).
- **Why the plain warp fails for obstacles:** the homography is only correct for points on the road. A car's body is
  above the road, so the warp stretches it into a long streak away from the camera (`naive_ipm_vs_scene_bev.png`,
  middle column).
- **Obstacles:** YOLO26 instance segmentation (1C). The bottom band of each mask is where the obstacle touches the
  road (tyres, feet), the one part of it that does lie on the ground plane. Back-projecting the left and right ends
  of that band through the homography gives the obstacle's distance and lateral extent in metres.
- **Footprint:**
  - The camera only sees the near face, so the extent away from the camera is a per-class assumption (car 4.5 m,
    truck 7 m, bus 11 m, person 0.6 m).
  - For vehicles, the lowest pixels are the rear tyres, which sit behind the bumper. A 0.9 m contact-to-near-face
    setback was measured on Town01 ground truth.
- **Occlusion:** the streak region of the plain warp is ground hidden behind the obstacle and is drawn grey
  (unknown).

## Accuracy against CARLA ground truth (Town01)

225 frames from 5 autopilot drives with up to 79 NPC vehicles. The ground truth is every NPC's bounding box
projected into the ego frame. Each estimate is matched to the vehicle whose lateral extent it overlaps and whose
near face is closest.

| range of true near face | matched | median distance error | median abs. distance error | median lateral-centre error | median width error |
|---|---:|---:|---:|---:|---:|
| 2-10 m  | 70 | -0.01 m | 0.06 m | -0.04 m | 0.18 m |
| 10-20 m | 32 | -0.32 m | 0.63 m | -0.06 m | -0.08 m |
| 20-30 m | 24 | +0.99 m | 1.16 m | -0.09 m | 0.46 m |
| 30-45 m | 13 | -2.73 m | 2.73 m | 0.08 m | -0.08 m |

- Before the 0.9 m setback the near-range bias was +0.89 m.
- Overall median relative distance error: 3.7% of range.
- 89% of the estimates match a real vehicle. 34% of the vehicles in the field of view are matched; most of the rest
  are occluded by other cars or too far away for YOLO at confidence 0.3.

## Limitations

- **Flat-ground assumption:** road grade or vehicle pitch shifts every distance estimate.
- **Distance resolution:** one image row covers ~0.3 m of ground at 10 m but ~2 m at 25 m, so the error grows
  quickly with range.
- **Truncated obstacles:** when an obstacle touches the bottom image edge, its real contact is below the frame and
  the estimate is an upper bound (`BEVObstacle.truncated`).
- **Footprint length** comes from the class. A van classified as "bus" is drawn 11 m long.
- **Hidden ground:** nothing is known about the road behind an obstacle. The BEV only covers 4.5-22.5 m ahead and
  +-12 m sideways.
