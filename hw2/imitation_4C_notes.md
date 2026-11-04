# Integrated pipeline: imitation learning (unit 4C)

Code: `hw2/imitation/`. Data and models: `module/data/hw2_imitation/` (git-ignored). Figures and run logs:
`report_figures/4C/`. Everything was collected, trained and evaluated on Town01.

## Setup

**Expert data** (`collect.py`)

- The expert is CARLA's traffic-manager autopilot on Town01, with:
  - no other traffic;
  - traffic lights and signs ignored;
  - the Town01 speed limit of 30 km/h;
  - no lane changes.

  This reduces the task to following the road, the same task the modular pipeline solves.
- 30 episodes of 75 s (1500 steps at 20 Hz), each from a different spawn point (seeds 100-129). Every other camera
  frame is saved (320 x 240, the benchmark camera), 22,500 frames in total.
- **Recovery data:** every 5 s the autopilot is switched off for 0.4 s and the car is steered off its line with a
  random offset (up to 0.3 steer). The autopilot then steers back. Forced steps, and frames whose 2 s label window
  touches one, are never used as labels. The recovery that follows is used, and it shows the networks how to get
  back to the lane, which plain expert driving never does.

**Labels** (`dataset.py`)

- **End-to-end:** the expert's next (steer, throttle, brake).
- **Waypoints:** the ego positions 0.5, 1.0, 1.5 and 2.0 s later, in the current vehicle frame (forward, right in
  metres), computed from the recorded poses.

**Inputs**

- **RGB:** the camera frame.
- **BEV:** the same frame warped by the 1A homography (`LaneDetection.front2bev`: 4.5-22.5 m ahead, +-12 m wide).

**Splits:** 27 episodes for training (11,185 usable frames) and the last 3 for validation (1,140). The split is by
episode: consecutive frames are near-duplicates, so a random split would leak them into validation.

**Network** (`network.py`): identical for all four variants.

- An ImageNet-pretrained ResNet-18 on the 240 x 320 image.
- The current speed goes through a small MLP and is concatenated with the image features.
- Head: 544 -> 256 -> output.
  - End-to-end: tanh for steer, sigmoid for throttle and brake.
  - Waypoints: 4 x 2 metres.
- Training: L1 loss, with steer weighted 2x in the end-to-end loss. AdamW, lr 3e-4 with cosine decay, batch 32,
  12 epochs, bf16 autocast. The epoch with the lowest validation loss is kept. About 14 s per epoch on the RTX
  4070 Ti.

**Controllers at test time** (`evaluate.py`)

- **End-to-end:** the network's output is applied directly. Gas and brake are made exclusive: the larger one wins.
- **Waypoint-based:** the predicted waypoints are converted into the lane detector's cut coordinates and followed
  with the 3B/3C Stanley controller (k = 3, D = 0.6). The target speed is the length of the predicted path divided by
  its 2 s horizon, and it is tracked with the 4B PID. The network decides where to go and how fast; the classic
  controllers execute it.
- **Modular baseline:** the 1A-3C pipeline on the detected path, with the 2C target speed capped at Town01's
  30 km/h (v_max = 30, v_min = 15), and the 4B PID.

## Offline results (validation episodes)

| model | best epoch | steer MAE | throttle MAE | brake MAE | waypoint ADE | final waypoint error |
|---|---:|---:|---:|---:|---:|---:|
| RGB, end-to-end | 12 | 0.028 | 0.148 | 0.006 | | |
| BEV, end-to-end | 7 | 0.035 | 0.152 | 0.006 | | |
| RGB, waypoints | 12 | | | | 1.05 m | 1.81 m |
| BEV, waypoints | 11 | | | | 0.80 m | 1.47 m |

- **For waypoints, BEV is clearly better: 24% lower ADE.**
  - Future positions are ground-plane geometry, and in the BEV a lateral offset or a curve is a linear image
    feature at a fixed metric scale.
  - From the RGB image the network has to learn the perspective mapping itself.
- **For end-to-end control, RGB is slightly better.**
  - The BEV discards everything outside 4.5-22.5 m ahead: the road right in front of the car, and the far view
    where upcoming curves and junctions are visible.
  - It also stretches distant pixels into blur.
  - BEV control also overfits sooner (best epoch 7).
- **Throttle errors are large for both inputs (~0.15).** The autopilot's throttle depends on its internal speed
  controller state rather than on the image. The waypoint labels avoid this: speed is implied by the spacing of the
  waypoints.
- **Junctions are ambiguous** (`offline_waypoints.png`, frame 250).
  - Without a route command the image doesn't say which way the expert will turn. Both models predict a turn where
    the expert went straight.
  - A regression network averages the possible branches when unsure, which is a path into the corner of the
    junction.

## Closed-loop results

The same 5 spawn points (seeds 1-5, none used for training) per agent; 1000 steps (50 s) each, or until the car
stood still for 10 s. Means over the 5 runs:

| agent | distance | distance before first lane departure | in lane | mean abs. offset | collisions | mean speed | stuck |
|---|---:|---:|---:|---:|---:|---:|---:|
| **RGB, end-to-end** | 400 m | **350 m** | 98.8% | **0.07 m** | 0.2 | 28.8 km/h | 0 / 5 |
| BEV, end-to-end | **416 m** | 331 m | **99.6%** | 0.09 m | 0.2 | 31.0 km/h | 1 / 5 |
| RGB, waypoints | 302 m | 92 m | 84.1% | 0.18 m | 0.6 | 22.9 km/h | 1 / 5 |
| BEV, waypoints | 297 m | 128 m | 89.7% | 0.20 m | 0.4 | 22.4 km/h | 1 / 5 |
| modular pipeline (1A-4B) | 118 m | 16 m | 48.5% | 0.38 m | 1.6 | 13.2 km/h | 4 / 5 |

(400 m in 50 s is about the maximum at 30 km/h after the start-up.)

### End-to-end vs. waypoint-based

**End-to-end won in closed loop, even though the waypoint models predict the future path to within ~1 m.**

1. **Its output is exactly the expert's behaviour, at the expert's rate.** The network copied the autopilot's
   steering, including how early it turns into a corner. Its smooth regression output gives the lowest steering
   jitter of all agents.
2. **The waypoint models are only as good as the hand-tuned controller that follows them.**
   - Stanley uses only the first waypoint (0.5 s ahead) and the first segment. That is about 3-4 m at these speeds,
     closer than the 4.5-13.5 m path it was tuned on in 3B. A 0.3 m error in that point is a large heading error.
   - The predicted waypoints jitter from frame to frame, just like the detected path in 3B. The car weaves within
     the lane (offset 0.18-0.20 m vs 0.07 m) and drifts out at the edges of turns. The first departure comes after
     ~100 m instead of ~350 m.
3. **Regression to the mean shortens the waypoints.**
   - Averaging over uncertain futures, for example accelerating vs. braking, or the branches of a junction, gives a
     shorter predicted path than the expert drove.
   - The derived target speed is therefore too low: 22-23 km/h instead of ~29.
4. **Shared failure.** Both waypoint models end seed 5 the same way: in a right turn they cut onto the sidewalk and
   hit a sign post (`closed_loop_failures.png`). This is the junction ambiguity above: the averaged path cuts the
   corner.

**Waypoints remain attractive.** They are interpretable (the path can be drawn and checked; see the
`closed_loop/*/` snapshots), and they separate "where to go" from "how to actuate". The latter can be swapped
between vehicles, and it is where the clear BEV advantage shows. With a controller designed for them, they would
likely close the gap: a pure-pursuit look-ahead point interpolated along all four waypoints, plus temporal smoothing
of the predictions.

### RGB vs. BEV

- **In closed loop the input mattered much less than the output type.**
  - The two end-to-end models are on par: BEV stays in lane slightly more, RGB stays closer to the centre. One
    stuck run (BEV, seed 4, a junction corner) is the main difference.
  - For waypoints, BEV's better offline accuracy shows as fewer lane departures (89.7% vs 84.1% in lane), but both
    suffer from the controller issues above.
- **BEV pros:**
  - metric, perspective-free road geometry, which helps the waypoint regression;
  - the same representation the modular pipeline plans in.
- **BEV cons:**
  - It is only correct for flat ground and a level camera. Pitching under braking distorts it (see 1F).
  - It throws away the near field (< 4.5 m) and the far view.
  - Objects above the ground (cars, posts) are smeared into long streaks.
- **RGB keeps all the information, but the network must learn the perspective.** That costs accuracy on geometric
  outputs with a small dataset like this one (~11k frames).

### All learned policies vs. the modular pipeline

Every learned policy drove 2.5-3.5x as far as the modular pipeline, and stayed in lane on 84-100% of steps instead of
48%. The modular pipeline is limited by its hand-designed lane detection (see the 3B notes: ~45% of boundaries
wrong, 0.4 m frame-to-frame jumps). The networks learned a far more robust notion of "the road" from 11k frames. The
learned policies also keep the expert's limitations: they ignore traffic lights and other vehicles, because the
expert data did.

## Limits of this comparison

- **One training run per variant, 5 evaluation routes each.** Differences of a few percent (e.g. the two end-to-end
  models) are within run-to-run noise.
- **No traffic or lights.** These policies are not ready for the benchmark routes, which have both.
- **No route command.** At junctions the policies choose a branch implicitly, and the waypoint models may average
  branches.
- **Python version for the benchmark.** The benchmark agent will likely run on the TA's CARLA 0.9.10 / Python 3.7
  setup (see the HW1 test kit). These modules use `from __future__ import annotations` so the annotations are not
  evaluated, but torchvision's weights API and bf16 autocast still need checking against that environment before
  2.5.

## Reproducing

```bash
cd hw2/imitation
python collect.py                                   # 30 episodes, ~15 min, ~5 GB
python train.py --input rgb --output control        # and bev/control, rgb/waypoints, bev/waypoints
python evaluate.py                                  # all agents, seeds 1-5; writes closed_loop.json + figures
python evaluate.py --agents bev_waypoints --seeds 3  # watch a single run
```
