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
Evaluation Layer  →  takes weight vector, runs simulation, returns {fitness, descriptors, raw stats}
Individual Layer  →  wraps weight vector + last evaluation results (shared across algorithms)
Algorithm Layer   →  MAP-Elites archive logic (swappable to MOEA/D without touching other layers)
```

### Network (Controller)

- **Topology:** Fixed feedforward MLP, `25-16-16-4` (~720 params, under 1000-parameter constraint)
- **Inputs (double-stacked):** current + previous timestep of [delta_position(2), angle(1), velocity(2), angular_velocity(1), thruster_angles(4), bias(1)] = 25 inputs total
- **Outputs:** 4 continuous thruster angle commands
- **Genome:** flat `float32` vector of all weights, direct encoding (no CPPN, no delta encoding)
- **Bias:** handled as a bias node in the input layer, not as a separate learned parameter

### MAP-Elites Archive

- **2D behavioral descriptor grid:**
  1. **Predictive error ratio** — mean dist to cursor k steps ahead / mean dist to cursor now (k calibrated empirically once simulation exists)
  2. **Overshoot ratio** — max overshoot past new trajectory / initial tracking error at direction change, averaged over all transient events
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
- Curriculum stage definitions (difficulty levels and transition criteria)
- Predictive error ratio lookahead `k` (calibrate against drone mechanical response time)
- Standardized evaluation trajectory set (straight line, figure-eight, sharp reversal, slow curve, hover)
