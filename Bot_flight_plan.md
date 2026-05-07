# Neuroevolutionary Structure Plan
## Continuous Control — Thruster-Based Drone Locomotion

---

## Problem Definition

**Task:** Train a drone controller to follow and hover at a moving mouse cursor in real-time, with potential anticipation of cursor movement.

**Reward structure:** Sparse + dense hybrid
- Dense: v_ratio component — current velocity toward target / maximum safe velocity (dynamically adjusted to distance from target)
- Sparse: target acquisition bonus
- Efficiency term: spawn distance / distance travelled
- Completion bonus scaled by spawn distance / distance travelled

**Environment:** Custom 2D physics simulation (to be built)
- Inputs: delta position, angle, velocity, angular velocity, current thruster angles, bias node
- Observation: double-stacked (current timestep + previous timestep of all inputs) to provide finite memory without recurrence
- Outputs: thruster angle commands (continuous)

---

## Point 1 — Target of Evolution

**Decision: Weights only, topology fixed**

- Genome is a flat float vector of all network weights
- Bias handled via a bias node in the input layer, not learned separately
- Topology is designed once before evolution begins and never changes
- Rationale: compatible with gradient hybridization, clean curriculum transfer between stages, sufficient signal from dense reward component, avoids speciation complexity of NEAT-style search

---

## Point 2 — Encoding Scheme

**Decision: Direct encoding, raw float weights**

- Genome = flat vector of raw float values, directly used as network weights
- No delta encoding (no base network to perturb around)
- No indirect encoding (no CPPN, no HyperNEAT)
- Rationale: gradient hybridization requires direct encoding, curriculum stage transfer is trivial with flat weight vectors, indirect encoding adds indirection that worsens sparse reward signal, direct encoding performs best for archive coverage in MAP-Elites locomotion benchmarks

---

## Point 3 — Topology Constraints

**Decision: Fixed feedforward MLP, no recurrence**

- Architecture: feedforward MLP, 25-16-16-4 (~720 parameters). Under 1000 parameter constraint satisfied.
- No recurrent connections — current kinematic state is fully observable, feedforward is sufficient
- Double-stacked inputs replace recurrence for finite memory — current and previous timestep of all sensor inputs concatenated into one flat observation vector
- If gait phase ambiguity emerges in practice, add previous timestep stack as additional inputs
- No subnetwork protection or module freezing at this stage
- Activation functions: to be decided at implementation time (tanh or ReLU standard choices for locomotion)

---

## Point 4 — Objective Formulation

**Decision: Decouple fitness from behavioral descriptors**

- Fitness function retains v_ratio dense component as backbone — provides signal on every timestep regardless of completion
- Completion bonus retained, scaled by spawn distance / distance travelled for efficiency
- Speed and precision preference removed from fitness — these become descriptor axes, not score components
- Fitness measures episode quality independently of where the solution falls on the speed/precision behavioral spectrum
- Curriculum controls spawn difficulty progression across stages — not a full reward redesign per stage
- Curriculum remains discrete stages, archive fills on easy conditions first then progressively harder spawns introduced

**Behavioral descriptor axes (finalized):**

**Primary axes (2D archive):**

1. **Predictive error ratio** — mean(distance from drone to cursor position k steps ahead) / mean(distance from drone to cursor current position), computed on moving-cursor segments only. Values below 1.0 indicate anticipatory behavior (drone is closer to where the cursor is going than where it currently is). Values above 1.0 indicate reactive/lagging behavior. k lookahead horizon to be calibrated empirically once simulation exists — should correspond roughly to the drone's mechanical response time.

2. **Overshoot ratio** — at each cursor direction change in evaluation trajectories, measure max distance the drone travels past the cursor's new trajectory line, divided by the initial tracking error at the moment of direction change. Averaged across all transient events in all evaluation runs. Value of zero means monotonic approach (never flies past target). High values indicate the controller consistently overshoots and corrects back. Requires a fitness floor — non-tracking controllers that never reach the target are excluded from descriptor computation and binned separately.

**Tertiary axis (for 3D archive expansion later):**

3. **Mean absolute body angle** — averaged across full episode on all evaluation runs. Low values indicate conservative upright flight posture prioritizing stability. High values indicate aggressive leaning prioritizing lateral agility at the cost of vertical stability. Maps to a genuine physical tradeoff in the drone dynamics.

---

## Point 5 — Diversity Mechanism

**Decision: MAP-Elites archive as primary diversity mechanism, with curiosity-weighted selection**

**Algorithm: Vanilla MAP-Elites as starting point, upgrade path to PGA-MAP-Elites**

- Archive is the primary diversity mechanism — grid forces behavioral diversity by construction
- Curiosity-weighted parent selection: cells that have been sampled heavily get lower selection weight, cells with mostly empty neighbors get higher weight
- Simple implementation: inverse sample count weighting per cell, or neighbor occupancy check on adjacent cells
- ISO-DD interpolation available as an exploration emitter arm (see Point 7)
- MOEA/D retained as backup algorithm — infrastructure designed for swappability (see Scalability)

**Why MAP-Elites over MOEA/D for this problem:**
- Behavioral space not yet well characterized — MAP-Elites discovers structure, MOEA/D assumes it
- Archive provides deployable repertoire for runtime controller selection based on cursor dynamics
- Cursor-following requires different behavioral modes (fast tracking vs precise hovering) — archive preserves both
- Direct encoding performs best for archive coverage in MAP-Elites locomotion benchmarks
- PGA-MAP-Elites outperforms existing methods on continuous control locomotion tasks including uncertain domains

---

## Point 6 — Selection & Population Structure

**Decision: Neutral drift replacement, batch evaluation**

- Archive replacement rule: neutral drift allowed — new candidate replaces cell occupant if fitness is equal to or greater than current occupant. Allows weight space exploration without fitness regression, helps escape local optima within cells
- No probabilistic replacement
- Evaluation protocol: each candidate evaluated on 4 runs. Fitness computed via weighted min aggregation across runs — penalizes catastrophic failure on any single run without being as harsh as pure min. Weights are fixed, not tuned during training.
- Incumbent re-evaluation: when a candidate challenges a cell occupant, the incumbent is also re-evaluated to prevent lucky evaluations from permanently holding cells
- Batch evaluation: N candidates evaluated per generation before archive update and bandit weight recomputation
- Batch size: multiple of CPU core count, suggested 20–64 range depending on simulation cost
- All proposals from all emitters pooled into one batch per generation
- Archive update happens once per batch after all results return
- Bandit weight update happens once per batch after archive update

---

## Point 7 — Variation Operators

**Decision: Four-arm emitter system managed by UCB1 bandit**

Each emitter is a class with a `propose(archive)` method returning a candidate weight vector. Emitters run in parallel during proposal generation (archive is read-only at this stage). All proposals simulate together in one batch. Results routed back to emitters by tag after simulation.

### Exploration Arms

**Arm 1 — Gaussian Exploration Emitter**
- Parent selection: curiosity-weighted (favors underexplored archive regions)
- Variation: Gaussian mutation with high adaptive sigma
- Sigma schedule: slow decay over training, remains large to maintain exploration pressure
- Stateless per candidate, trivially parallelizable

**Arm 2 — ISO-DD Interpolation Emitter**
- Parent selection: two parents from distant archive cells
- Variation: directional interpolation between parent weight vectors plus small perpendicular perturbation
- Systematically explores behavioral space between existing solutions
- Targets cells in behavioral regions that random mutation struggles to reach
- Stateless per candidate, trivially parallelizable

### Exploitation Arms

**Arm 3 — CMA-ES Emitter**
- Maintains own mean vector and covariance matrix centered on a high-performing archive elite
- Samples candidates from learned distribution, updates covariance based on which candidates successfully entered archive
- Learns local geometry of fitness landscape around target elite — biases proposals toward directions that historically produce archive improvements
- Model-free — does not require critic, uses ground truth archive outcomes directly
- Internal state: mean vector and covariance matrix per emitter instance
- Candidates tagged for tracking — after batch evaluation, CMA-ES receives only its own tagged results for covariance update before next round
- Covariance update is fast bookkeeping, does not require additional simulation runs
- Scalability note: covariance update has sequential dependency (evaluate → update → propose) but evaluation itself parallelizes normally

**Arm 4 — Gradient Emitter**
- Requires critic network (separate from controller) trained on replay buffer data
- Critic estimates expected cumulative dense reward from state-action pairs
- Replay buffer: circular queue of (state, action, reward, next state) tuples collected during all simulation episodes, capped at ~100k entries
- Critic trained offline between evaluation batches using TD learning on replay buffer samples
- Emitter operation: takes archive elite, runs forward pass on states sampled from replay buffer, computes policy gradient via critic, takes gradient step on elite weights, proposes result as candidate
- Fast exploitation in well-explored archive regions where critic is accurate
- Unreliable in sparse archive regions where critic has insufficient data — bandit naturally handles this by reducing arm budget when improvement rate drops
- Accuracy gate: gradient emitter is not activated until critic achieves ~85% accuracy, measured as rank correlation on a held-out validation set of (state, action, reward) tuples. This prevents wasting bandit exploration budget on a broken arm during early training.
- Dependent on critic quality — complementary to CMA-ES which is model-free and robust

### Bandit Manager

- Algorithm: UCB1
- Each arm tracks: proposals made, successful archive entries produced
- UCB1 selects arm with highest upper confidence bound on improvement rate
- Budget per arm per batch = bandit allocation × total batch size (unequal arm sizes, not fixed split)
- Bandit weights reset or decayed at curriculum stage transitions to prevent stale allocation

---

## Point 8 — Gradient Hybridization

**Decision: Gradient emitter as exploitation arm (see Point 7 Arm 4)**

- Full PGA-style gradient emitter included as fourth bandit arm
- Critic network + replay buffer infrastructure built into evaluation loop from the start, even before gradient emitter is activated
- Lamarckian shortcut (gradient steps post-evaluation without critic) available as simpler interim option during development
- Gradient emitter added last in implementation sequence, after vanilla MAP-Elites system validated

---

## Point 9 — Scalability

**Hardware context:** Oracle Cloud ARM Flex A1 — CPU only, up to 4 cores, no GPU

**Primary bottleneck:** Simulation throughput, not network inference or archive management

**Parallelization strategy:**
- Emitter proposal generation: all four emitters run in parallel (read-only archive access, no race conditions)
- Simulation: multiprocessing pool across 4 ARM cores, one episode per worker
- Batch size should be multiple of core count
- Post-simulation bookkeeping (archive update, CMA-ES covariance, bandit weights, critic training) sequential but fast

**Network inference:** Batched NumPy matrix operations for forward passes across all candidates simultaneously. Under 1000 parameters — NumPy BLAS is sufficient, PyTorch overhead not worth it on ARM CPU for this scale. JAX with JIT compilation is an option for better CPU throughput if needed.

**GPU note:** No GPU on Oracle ARM instance. Batched CPU inference is the strategy. If local machine has GPU, CPU-GPU transfer overhead likely exceeds compute savings for under 1000 parameter networks — do not design around GPU acceleration for this parameter scale.

**Archive resolution scaling:**
- Consider coarse grid early, finer resolution as curriculum advances and behaviors become more refined
- Changing resolution mid-training requires remapping elites to new cells — plan for this before implementation
- Too coarse loses behavioral distinction, too fine keeps cells empty too long

**Curriculum stage transition policies (all need explicit decisions at implementation time):**
- Archive at transition: options are keep all elites + re-evaluate, keep without re-evaluation, or partial reset
- Replay buffer staleness: sliding window or curriculum-tagged transitions with recency weighting
- Bandit weight staleness: reset or decay arm weights at each stage transition
- Single archive vs per-curriculum archive: affects stepping stone carryover between stages

**MOEA/D swappability:**
Infrastructure layered in three clean separable levels:
1. Evaluation layer — takes weight vector, runs simulation, returns results dict with fitness components, descriptor values, raw episode stats. Neither algorithm touches this directly.
2. Individual/population layer — Individual class wrapping weight vector and last evaluation results. Shared between MAP-Elites and MOEA/D.
3. Algorithm layer — swappable. MAP-Elites archive logic here. MOEA/D population and decomposition logic would also live here. Both call down into identical evaluation layer.

Swapping to MOEA/D requires replacing only the algorithm layer. Emitter variation operators are reusable in MOEA/D context.

---

## Point 10 — Algorithm Backbone

**Decision: Curriculum-managed multi-emitter MAP-Elites with critic-guided gradient hybridization**

Closest published analogues: PGA-MAP-Elites + QDax multi-emitter architecture + curriculum layer. No single paper describes this exact combination — validated components assembled into a coherent system.

### Full Loop Structure

**Outer loop — Curriculum Manager**
- Maintains current difficulty stage (spawn distance, spawn angle range, cursor movement speed)
- Monitors archive coverage and mean fitness metrics to trigger stage transitions
- Executes transition policy: archive handling, bandit weight reset/decay, replay buffer staleness management
- Does not modify reward function or descriptor axes between stages

**Inner loop — MAP-Elites with Bandit Emitters**
1. Bandit allocates batch budget across four emitters via UCB1
2. All four emitters generate proposals in parallel (read-only archive access)
3. All proposals tagged by emitter, pooled into one batch
4. Batch simulates in parallel across CPU cores via multiprocessing pool
5. Archive updates with neutral drift replacement rule for all results
6. CMA-ES receives its tagged results, updates covariance matrix
7. Critic trains on new replay buffer entries collected during batch simulation
8. Bandit updates arm weights based on archive improvements attributable to each emitter
9. Curiosity weights updated per cell based on sample counts and neighbor occupancy
10. Repeat

### Implementation Sequence (do not deviate from this order)

1. Build simulation environment
2. Implement vanilla MAP-Elites with Gaussian exploration emitter only
3. Validate descriptor axes and fitness function — check archive fills meaningfully
4. Add curiosity-weighted selection to Gaussian emitter
5. Add ISO-DD interpolation emitter
6. Add UCB1 bandit manager
7. Add CMA-ES emitter
8. Build replay buffer infrastructure into evaluation loop
9. Train critic network
10. Add gradient emitter as fourth arm
11. Add curriculum manager on top of validated inner loop

---

## Open Items (must be resolved before implementation of archive)

1. **Archive grid resolution** — depends on descriptor axis ranges, which need empirical calibration once simulation exists.

2. **Curriculum stage definitions** — specific spawn difficulty levels and transition criteria not yet defined.

3. **Predictive error ratio k value** — lookahead horizon must be calibrated empirically against drone mechanical response time once simulation is built.

4. **Evaluation trajectory design** — standardized cursor trajectories for evaluation not yet defined. Candidates: straight line, figure-eight, sharp reversal, slow curve, stationary hover. Descriptor validity depends on consistent evaluation conditions across all candidates.
