# Longitudinal control notes (units 4A, 4B)

All tests ran on Town01. Figures and per-run logs are in `report_figures/4B/` (PNGs git-ignored).

## 4A: PID controller

`LongitudinalController.PID_step(speed, target_speed)` implements eqs. 5 and 6:

- e(t) = v_target - v(t)
- u(t) = K_p e(t) + K_d [e(t) - e(t-1)] + K_i sum e

`control()` maps u to the pedals as in eq. 7: gas = u if u >= 0, brake = -u otherwise. Each is clipped to 0.8 as in
the template.

- **Units:** speeds are in km/h, the unit of the 2C target speed. CARLA's m/s are converted in the tests.
- **Anti-windup:** the error sum is clipped to +-`max_error_sum`. Without a bound, the sum keeps growing while the
  car cannot follow, for example while accelerating from standstill at full gas. It must then be paid back by an
  equally long error of the opposite sign, which shows up as a long overshoot. The bound is chosen so that
  `K_i * max_error_sum` equals the largest pedal value (0.8): the integral term can hold any steady pedal position,
  but cannot wind up beyond what the pedals can deliver.

## 4B: parameter search

`test_longitudinal_control.py` runs the full pipeline: Stanley steering (k = 3, D = 0.6) and PID gas/brake.
It plots speed against target speed and prints step-response statistics at the end.

- **Ground-truth path:** tuning used `--path ground-truth` (the map's lane centre, see the 3B notes). This keeps the
  car on the road, so the speed control can be judged on its own. On the detected path the car leaves the road
  within ~20 m.
- **Step target:** tuning also used `--target steps`, a target that steps through 20, 40, 15, 35 and 25 km/h every
  8 s. The 2C target speed is almost constant at 60 km/h (see the 2C notes), so it has no steps to tune against,
  and 60 km/h is twice Town01's speed limit.
- **Procedure:** as the assignment suggests, the P term was tuned first, then one term at a time was added. Same
  spawn point (seed 3) for every run; 800 steps (40 s).
- **Metrics:** overshoot is the largest excursion beyond the new target, as a share of the step. Settling is the
  time until the speed stays within +-2 km/h of the target. Acceleration is from the speed change between steps.
  The first second is excluded, because the car is spawned slightly above the road and bounces.

### 1. P only (`p_only_kp*.png`)

| K_p | RMS error | settled steps | overshoot |
|---:|---:|---:|---:|
| 0.02 | 18.7 km/h | 0 / 5 | 76% |
| 0.05 | 9.7 km/h | 0 / 5 | 12% |
| 0.1 | 6.7 km/h | 0 / 5 | 28% |
| **0.2** | 5.5 km/h | 2 / 5 | 23% |
| 0.4 | 5.3 km/h | 5 / 5 | 23% |

- A P controller needs a non-zero error to produce gas, so it always settles below the target. The MKZ needs
  ~0.4 gas just to hold 20-40 km/h.
  - With K_p = 0.02 (template 0.01) it never gets close.
  - At K_p = 0.2 it holds 18 instead of 20 km/h, and 38 instead of 40 km/h.
- K_p = 0.4 halves that offset, but the speed starts oscillating around 25 km/h, where CARLA's automatic gearbox
  shifts. **K_p = 0.2** was kept, and the remaining offset was left to the integral term.
- The large "overshoot" after the 40 -> 15 km/h step is an undershoot. The error of -25 km/h saturates the brake at
  0.8, and the car drops to 7 km/h before the P term releases it.

### 2. Adding I (`ki_sweep.png`, K_p = 0.2, anti-windup bound 0.8 / K_i)

| K_i | RMS error | settled steps | mean settling time | overshoot |
|---:|---:|---:|---:|---:|
| 0 | 5.5 km/h | 2 / 5 | 5.0 s | 23% |
| 0.002 | 4.9 km/h | 5 / 5 | 2.2 s | 29% |
| 0.005 | 4.9 km/h | 5 / 5 | 2.2 s | 33% |
| **0.01** | 4.8 km/h | 5 / 5 | 1.7 s | 34% |
| 0.02 | 4.8 km/h | 5 / 5 | 2.6 s | 35% |

- Any K_i removes the steady-state offset.
- K_i = 0.002 needs a sum of 400 to supply 0.8 gas, so the offset only shrinks slowly. It also overshoots on
  40 km/h, because the large bound lets the sum wind up during the acceleration.
- K_i = 0.02 starts to oscillate.
- **K_i = 0.01** (bound 80) settles fastest. A +-3 km/h oscillation around 20 km/h remains (gear shifts).

### 3. Adding D (`kd_sweep.png`, K_p = 0.2, K_i = 0.01)

| K_d | RMS error | mean settling time | overshoot | max. deceleration |
|---:|---:|---:|---:|---:|
| 0 | 4.8 km/h | 1.7 s | 34% | 26 m/s^2 |
| 0.1 | 4.8 km/h | 1.7 s | 31% | 26 m/s^2 |
| 0.3 | 4.7 km/h | 1.5 s | 26% | 26 m/s^2 |
| 0.6 | 4.7 km/h | 1.4 s | 19% | 26 m/s^2 |
| **1.0** | 4.8 km/h | 1.6 s | 13% | 17 m/s^2 |

- The D term counters fast speed changes:
  - It removes the oscillation around 20 km/h.
  - After a downward step it releases the brake as soon as the car decelerates hard. The undershoot to 5-7 km/h
    becomes 11 km/h, and the deceleration peak drops by a third.
- The derivative acts on the error, so a target step also gives a one-step "derivative kick" of K_d times the step
  size. With these step sizes the pedals are already saturated by the P term, so the kick changes nothing.
- **K_d = 1.0** was chosen. It is the smoothest response (lowest overshoot and acceleration peaks), and its settling
  time is within 0.2 s of the fastest. The smooth response is the point of the assignment's warning about strong
  acceleration after tight corners: gentle speed changes keep the tyres within their grip.

### Final parameters

| K_p | K_i | K_d | anti-windup bound | max gas / brake |
|---:|---:|---:|---:|---:|
| 0.2 | 0.01 | 1.0 | 80 km/h * steps (integral term at most 0.8) | 0.8 |

### Validation on other spawn points and on the predicted target (`final_validation_*.png`, `predicted_target.png`)

- **Step target, seed 2:** RMS error 4.6 km/h, all steps settled (mean 1.5 s), overshoot 13%. Same as on the tuning
  seed.
- **Step target, seed 1:** the first three steps settle in about 1 s. The car then hits a wall at 15 km/h at a
  T-junction where the ground-truth path ends (`final_validation_s1_wall.png`). This is a path problem, not a
  speed-control problem.
- **Predicted target, v_max = 60 (as in the template test):**
  - The speed follows the 60 km/h target, and dips with it to ~48 km/h at each corner. The PID lags the dips by
    about 1 s, and accelerates back to 61 km/h after the corner (small overshoot, no wheel spin).
  - After 34 s the car leaves the road in a 90 deg junction turn at 50 km/h.
  - 2C's target speed only drops to ~52 km/h for such corners. Town01's corners need about 20-30 km/h; v_max = 60 is
    too fast for the town.
- **Predicted target, v_max = 30:** with v_min = 30 too, the target is constant. The car holds 30.0 +- 0.3 km/h for
  the whole run. With P only (K_p = 0.2) it holds 27.8 km/h, 2.2 km/h low, which is the offset the I term removes.
