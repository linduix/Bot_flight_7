# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**Goal:** Train a drone controller (neuroevolution, no gradient by default) to follow and hover at a moving mouse cursor in real-time, in a custom 2D physics simulation.

Full design decisions are documented in [`Bot_flight_plan.md`](Bot_flight_plan.md).

---

## Implementation Sequence

Do not skip steps or reorder — each stage validates the components needed by the next:

1. Build the 2D physics simulation environment
2. Implement vanilla MAP-Elites with Gaussian exploration emitter only
3. Validate descriptor axes and fitness function (archive fills meaningfully)
4. Add curiosity-weighted parent selection to the Gaussian emitter
5. Add ISO-DD interpolation emitter
6. Add UCB1 bandit manager
7. Add CMA-ES emitter
8. Build replay buffer infrastructure into the evaluation loop
9. Train critic network
10. Add gradient emitter as fourth bandit arm
11. Add curriculum manager on top of the validated inner loop

---

## Architecture

### Three-Layer Separation (must remain clean for MOEA/D swappability)

```
Evaluation Layer  →  takes genome (weights + biases), runs simulation, returns {fitness, descriptors, raw stats}
Individual Layer  →  wraps genome + last evaluation results (shared across algorithms)
Algorithm Layer   →  MAP-Elites archive logic (swappable to MOEA/D without touching other layers)
```

### Simulation

- **Implementation:** Custom 2D physics, batched JAX (`jit` + `vmap` + `lax.scan`)
- **Target chain (per seed):** 5 waypoints, first at (0,0). Remaining 4 at random angles + random distances normalized to sum to fixed `total_chain_length`.
- **Segment times (per seed):** 4 random durations normalized to sum to 80% of episode time `T`. Remaining 20% is the approach phase before touch.
- **Spawn (per seed):** always at origin (waypoint 0). Geometric variety comes from chain direction randomization, not spawn-location randomization. Same spawn for all drones on that seed (fairness).
- **Touch trigger:** drone within `touch_radius` of target → chain movement begins. One-way (cannot un-trigger). Not scored — only gates chain.
- **Chain motion:** linear interpolation between waypoints based on post-touch timer. Holds at final waypoint until `T`.
- **Initial drone state (per seed, randomized):** angle ∈ ±15° uniform, angular_velocity ∈ ±0.5 rad/s uniform, velocity in small isotropic ball (≤ ~10% of max), position at origin, t1/t2 angles zero. Same init for all drones on that seed (fairness). Prevents overfit to rest-upright starts and forces learned recovery transients.
- **Noise:** observation noise (Gaussian, ~1–2% of per-input scale, per channel per tick) added only after vanilla MAP-Elites validates noise-free. No action/dynamics noise at this stage.
- **Drone state (complex array, length 6):** position (x+iy), velocity (vx+ivy), angle, angular_velocity, t1_angle, t2_angle. Last 4 stored as complex with zero imag for uniform dtype.
- **Action (MLP outputs, 4 floats):** t1_thrust, t2_thrust (clipped to ≥0, applied immediately), t1_rot_rate, t2_rot_rate (∈ [-1, +1], gimbal rotates at `rotation_speed * rate` per tick; gimbal angle clipped to ±60°). Rate-command gimbal control, not position-command — network must output zero to hold a gimbal angle.
- **Fitness per tick:** `dt / (1 + distance_to_target)`. Multiplied by 0.01 pre-touch.
- **Episode fitness:** sum over all ticks. **Drone fitness:** mean across all S seeds.
- **Batching:** P drones × S seeds flattened to P×S parallel episodes. Chains pre-generated in NumPy, passed as static arrays. `lax.scan` over time, `vmap` over episodes.

### Network (Controller)

- **Topology:** Fixed feedforward MLP, `11-12-9-6-5-4` (32 hidden nodes, ~380 params, under 1000-parameter constraint). Pyramid narrowing informed by prior NEAT run.
- **Inputs (11 total):**
  - delta_position_now (2, body frame)
  - delta_position_prev (2, body frame) — only history, lets network infer target motion
  - velocity (2, body frame)
  - sin(angle), cos(angle) (2, world frame, gravity reference)
  - angular_velocity (1)
  - t1_angle, t2_angle (2, body frame, raw radians — clamped ±60° so no wrap)
- **Frame split:** angle world-frame (gravity reference); delta_position and velocity body-frame (rotation-invariant policy). Convert world→body via `value * exp(-1j * angle)`.
- **Outputs:** thruster_angles(2) + thrust_magnitudes(2) = 4 continuous outputs
- **Genome:** flat `float32` vector of all weights and per-node biases concatenated, direct encoding (no CPPN, no delta encoding). Bias vector length = 36 (one per non-input node).
- **Bias:** per-node scalar bias on every hidden and output node, stored as a separate bias vector in the genome alongside the weight matrices

### MAP-Elites Archive

- **2D behavioral descriptor grid:**
  1. **Mean absolute angular velocity** — `mean(|ang_vel|)` across all ticks (rad/s). Low = stable posture, high = aggressive spinning. Bounds: `[0, 10]` rad/s with the top bin acting as an overflow catch-all for anything ≥ 10. Calibrate after first batch.
  2. **Mean thrust saturation** — fraction of ticks where `max(t1, t2) > 0.9`. Strictly in `[0, 1]` by construction. Low = gentle control, high = bang-bang control.
- **Tertiary axis** (future 3D expansion): mean absolute body angle across episode
- **Replacement rule:** neutral drift — new candidate replaces occupant if fitness ≥ current occupant
- **Grid resolution:** TBD after empirical calibration of descriptor ranges

### Fitness Function

- Dense: `v_ratio` = current velocity toward target / max safe velocity (scaled by distance)
- Sparse: target acquisition bonus
- Efficiency: spawn distance / distance travelled
- Completion bonus scaled by spawn distance / distance travelled
- Evaluated on **4 runs per candidate**, aggregated via weighted min (penalizes catastrophic failures)
- Incumbent is re-evaluated when challenged (prevents lucky incumbent holdover)

### Four-Arm Emitter System (UCB1 bandit)

| Arm | Type | State |
|-----|------|-------|
| 1 | Gaussian Exploration — curiosity-weighted parent selection, high adaptive sigma | Stateless |
| 2 | ISO-DD Interpolation — two distant parents, directional interp + perpendicular noise | Stateless |
| 3 | CMA-ES — maintains mean + covariance around a high-performing elite | Per-emitter mean + covariance |
| 4 | Gradient Emitter — policy gradient via critic, activated only after critic ≥ 85% rank correlation | Critic network + replay buffer |

- Bandit allocates batch budget via UCB1 (unequal arm sizes, not fixed split)
- Bandit weights reset or decayed at curriculum stage transitions
- Gradient emitter activation gate: wait until critic accuracy ≥ 85% rank correlation on held-out validation set

### Replay Buffer

- Circular queue of `(state, action, reward, next_state)` tuples, ~100k cap
- Collected from all simulation episodes
- Critic trained offline between batches using TD learning on buffer samples
- Infrastructure built in at Step 8, before gradient emitter is added

### Curriculum Manager (outermost loop)

- Controls spawn difficulty: spawn distance, spawn angle range, cursor movement speed
- Monitors archive coverage + mean fitness to trigger stage transitions
- Does NOT modify reward function or descriptor axes between stages
- Stage transition decisions (all TBD at implementation time):
  - Archive handling: keep all / keep without re-eval / partial reset
  - Replay buffer: sliding window vs curriculum-tagged recency weighting
  - Bandit weights: reset vs decay

---

## Parallelization

- **Hardware:** Oracle Cloud ARM Flex A1, 4 cores, CPU only
- **Simulation:** `multiprocessing.Pool` across 4 cores, one episode per worker
- **Emitter proposals:** all four emitters generate in parallel (archive read-only during this phase)
- **Network inference:** batched NumPy matrix ops — no PyTorch/JAX required at <1000 params (consider JAX JIT if throughput becomes a bottleneck)
- **Batch size:** multiple of 4 (core count); suggested 20–64 range

---

## Open Items (resolve before implementing the archive)

- Archive grid resolution (requires empirical descriptor range calibration from simulation)
- Angular velocity descriptor upper bound (calibrate empirically; saturation axis is already `[0, 1]`)
- Curriculum stage definitions (difficulty levels and transition criteria)
- Standardized evaluation trajectory set (straight line, figure-eight, sharp reversal, slow curve, hover)
