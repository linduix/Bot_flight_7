# System Design — High Level Flow

## Core Philosophy
- Swap any component without touching others
- Loop is algorithm and simulator agnostic
- Each component owns its own state and stats
- Simulator only called from the main loop, never from inside algorithm

---

## Component Responsibilities

### Individual
Plain data carrier. No logic.
```
Individual:
    weights      # set at proposal, never changes
    fitness      # filled by simulator
    descriptors  # filled by simulator — determines archive cell
    stats        # filled by simulator — raw episode data
    tag          # which emitter proposed this
```

### Simulator
Black box. Takes weights + trial pool + config, returns filled Individuals.
- Stochastic: runs N trials internally, aggregates, returns one result per Individual
- Deterministic: runs once, returns result
- The loop never knows or cares which it is

### Algorithm
Black box. Two responsibilities only:
- `propose(n, config)` — return n weight vectors to evaluate
- `update(evaluated)` — receive results, update internal state

Internally (MAP-Elites specific, loop never sees this):
- Archive grid management
- Emitter coordination (Gaussian, ISO-DD, CMA-ES, Gradient)
- Bandit budget allocation (UCB1)
- Curiosity weighting
- Archive cell replacement (neutral drift)

### Trial Pool
Fixed set of cursor trajectories used to evaluate all candidates within a curriculum stage.
- All candidates in a pool period face identical conditions → fitness is directly comparable
- Pool only rotates when curriculum advances → no drift within a stage
- At rotation: all incumbents re-evaluated under new pool before next generation starts

### Curriculum Config
Passive config object. Controls difficulty:
- Spawn distance
- Spawn angle range  
- Cursor movement speed

Owned and mutated by the loop. Not a separate orchestration layer.

### Checkpointing
Two save points:
- **Rolling checkpoint** — saves every N generations, overwrites previous. Crash recovery.
- **Best checkpoint** — saves when global best fitness is beaten. Never overwritten.

Both store full state: algorithm internals, trial pool, config, generation, best_fitness.

### Event Emission
Loop emits one dict per generation containing raw stats from all components.
No formatting, no filtering — listener decides what to do with it.
Listener is fully replaceable without touching the loop.

---

## Layer Structure

```
run()                          ← the loop, coordinates everything
├── Algorithm                  ← swap here for MOEA/D etc.
│   ├── Archive
│   ├── Bandit
│   └── Emitters
├── Simulator                  ← swap here for different physics
└── Individual                 ← shared data contract between all layers
```

---

## High Level Flow

```
setup:
    simulator  = Simulator()
    algorithm  = Algorithm()
    trial_pool = generate_trial_pool(config)

run(algorithm, simulator, emit_fn):
    load rolling_checkpoint if exists
    generation   = 0
    best_fitness = -inf

    loop:
        batch,      propose_stats = algorithm.propose(n, config)
        evaluated,  sim_stats     = simulator.evaluate(batch, trial_pool, config)
        update_stats              = algorithm.update(evaluated)
        config,     stage_stats   = advance_stage(config, evaluated)

        if curriculum_advanced(stage_stats):
            trial_pool               = generate_trial_pool(config)
            revalidated, reval_stats = simulator.evaluate(algorithm.all_elites(), trial_pool, config)
            algorithm.revalidate(revalidated)

        emit_fn({
            generation,
            propose_stats,
            sim_stats,
            update_stats,
            stage_stats,
            reval_stats,   # only present when curriculum advanced
            config
        })

        best_fitness = checkpoint(evaluated, generation, best_fitness, N)

        if stop_condition:
            break

        generation += 1
```

---

## Key Design Decisions & Why

**Why pool-based trial evaluation?**
Fitness is only comparable if evaluated under the same conditions. A fixed pool per curriculum stage guarantees this. Per-challenge incumbent re-evaluation breaks separation of concerns and is harder to parallelize.

**Why tie pool rotation to curriculum transitions?**
If pool rotates on its own schedule while curriculum advances, candidates at the start of a pool period face easier conditions than candidates at the end — fitness drifts within the period. Tying them together eliminates this.

**Why does algorithm get simulator at construction?**
Originally considered but rejected — simulator should only be called from the loop. Incumbent re-evaluation is instead handled via the pool approach, so algorithm never needs a simulator reference.

**Why is curriculum a config object and not a layer?**
Curriculum is just difficulty settings that change over time. It has no orchestration logic of its own. Promoting it to a full layer adds indirection with no benefit.

**Why are stats returned from each step rather than pulled?**
Keeps the loop agnostic. Each component knows what it wants to expose. The loop collects and forwards. Adding or removing metrics never requires touching the loop.

**Why two checkpoint types?**
Rolling checkpoint protects against crashes — you lose at most N generations.
Best checkpoint protects your best result independently — a later crash or degradation can never destroy it.

---

## Open Decisions (not yet resolved)

- Stop condition: max generations, fitness threshold, time limit, or manual?
- Rolling checkpoint interval N: depends on generation cost once simulation is built
- Trial pool size and composition: how many trajectories, what types
- What `all_elites()` returns at curriculum transition and whether held buffer candidates are included
