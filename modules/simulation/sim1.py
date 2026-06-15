from modules.individual import Individual
from modules.batch_brain import brain
from multiprocessing.pool import Pool
import numpy as np
import tomllib
import os


# Drone constants
config_path = 'config.toml'
def get_drone_conf(path) -> 'dict':
    with open(path, 'rb') as f:
        config = tomllib.load(f)
        drone_config = config['drone']
    drone = {
        'M'                 : drone_config['mass'],
        'I'                 : drone_config['inertia'],
        'width'             : drone_config['width'],
        'height'            : drone_config['height'],
        'G'                 : drone_config['gravity'],
        'th_offset'         : drone_config['thruster_offset'],
        'th_rotation_speed' : drone_config['thruster_rotation_speed'],
        'th_max_angle'      : drone_config['thruster_max_angle'],
        'th_force'          : drone_config['thruster_force'],
        # convert throttle_rate (seconds for full sweep) -> rate (1/s)
        'th_actuation_rate' : 1.0 / drone_config['throttle_rate'],
    }
    return drone

# physics state
# 0. x + i*y
# 1. vx + i*vx
# 2. angle
# 3. angular vel
# 4. t1 angle
# 5. t2 angle
# 6. t1 thrust level + i * t2 thrust level (each clipped to [0, 1], fraction of th_force)
state = [0+0j, 0+0j, 0, 0, 0, 0, 0+0j]

# physics state matrix + actions matrix -> updated state
def physics_update(dt, state: np.ndarray, actions: np.ndarray, drone_conf: dict):
    assert state.ndim   == 3, f"state must be 3D (n, state, S), got shape {state.shape}"
    assert actions.ndim == 3, f"actions must be 3D (n, action, S), got shape {actions.shape}"

    # process actions — target-command. throttle outputs sigmoid [0, 1] are target thrust
    # fractions; gimbal outputs tanh [-1, 1] are target angles scaled by max_angle.
    t1_cmd, t2_cmd, rot1, rot2 = actions[:, 0, :], actions[:, 1, :], actions[:, 2, :], actions[:, 3, :]

    # GIMBAL — target-command. tanh [-1, 1] -> target angle [-max_angle, +max_angle].
    # actuator slews toward target at rotation_speed.
    rotation_speed = float(np.deg2rad(drone_conf['th_rotation_speed']))
    max_angle      = np.deg2rad(drone_conf['th_max_angle'])
    max_delta_rot  = rotation_speed * dt
    t1_ang_target  = rot1 * max_angle
    t2_ang_target  = rot2 * max_angle
    cur_t1_ang     = state[:, 4, :].real + np.clip(t1_ang_target - state[:, 4, :].real, -max_delta_rot, max_delta_rot)
    cur_t2_ang     = state[:, 5, :].real + np.clip(t2_ang_target - state[:, 5, :].real, -max_delta_rot, max_delta_rot)
    state[:, 4, :] = np.clip(cur_t1_ang, -max_angle, max_angle) # type:ignore
    state[:, 5, :] = np.clip(cur_t2_ang, -max_angle, max_angle) # type:ignore

    # THROTTLE — target-command. sigmoid [0, 1] IS the target thrust fraction; actuator
    # slews toward target at actuation_rate. state[:, 6] packs both: real=t1, imag=t2.
    actuation_rate = drone_conf['th_actuation_rate']
    max_delta_thr  = actuation_rate * dt
    cur_t1 = state[:, 6, :].real + np.clip(t1_cmd - state[:, 6, :].real, -max_delta_thr, max_delta_thr)
    cur_t2 = state[:, 6, :].imag + np.clip(t2_cmd - state[:, 6, :].imag, -max_delta_thr, max_delta_thr)
    cur_t1 = np.clip(cur_t1, 0.0, 1.0)
    cur_t2 = np.clip(cur_t2, 0.0, 1.0)
    state[:, 6, :] = cur_t1 + 1j * cur_t2

    # THRUST
    # magnitude
    thrust1 = cur_t1 * drone_conf['th_force']
    thrust2 = cur_t2 * drone_conf['th_force']

    # vector
    thrust1dir: np.ndarray = 1j * np.exp(1j * state[:, 4, :])
    thrust2dir: np.ndarray = 1j * np.exp(1j * state[:, 5, :])
    F1 = thrust1 * thrust1dir
    F2 = thrust2 * thrust2dir
    F = F1 + F2

    # TORQUE
    # magnitude
    tau1: np.ndarray = -drone_conf['th_offset'] * F1.imag
    tau2: np.ndarray =  drone_conf['th_offset'] * F2.imag
    T = tau1 + tau2

    # update physics (world frame) -------------------------------
    # rotate drone
    ang_acc   = T / drone_conf['I']
    state[:, 3, :] += ang_acc * dt        # ang vel
    state[:, 2, :] += state[:, 3, :] * dt # ang rotation

    # translation
    F_world = F * np.exp(1j * state[:, 2, :])
    acc: np.ndarray = F_world / drone_conf['M'] - 9.81j

    vel: np.ndarray = acc * dt
    state[:, 1, :] += vel

    trans: np.ndarray = state[:, 1, :] * dt
    state[:, 0, :] += trans

    return state

def gen_target_chain(length, limit, dt, rng, S, base_S=16) -> np.ndarray:
    # purpose: to map every tick in sim to the position of target after touch.
    # all rng draws are sized to `base_S` (default 16, matching sim1.py's training S) so
    # the rng state advances identically regardless of the actual S requested. trials
    # 0..S-1 are then byte-identical across callers that request different S values
    # with the same seed (e.g., training S=16, visual S=4).
    assert S <= base_S, f"S ({S}) must be <= base_S ({base_S})"

    n_segments = 5
    n_points   = n_segments + 1   # origin -> wp1 + motion segments

    # path length + speed -------------
    alpha = 3.0
    concentration = 15
    lengths = rng.dirichlet([alpha] * n_points, size=base_S)      # (base_S, segments + 1)
    alphas  = lengths[:, 1:] * concentration   # NOT including origin -> wp1
    g       = rng.gamma(alphas, 1.0)
    times   = g / g.sum(axis=1, keepdims=True)

    # path angles ---------------------
    maneuvers = {
        'straight': dict(mu=0.0,        kappa=4.0, weight=0.4),
        'corner':   dict(mu=np.pi / 2,  kappa=3.0, weight=0.4),
        'reversal': dict(mu=np.pi,      kappa=3.0, weight=0.2),
    }

    weights = np.array([m['weight'] for m in maneuvers.values()])
    mus     = np.array([m['mu']     for m in maneuvers.values()])
    kappas  = np.array([m['kappa']  for m in maneuvers.values()])

    # pick a maneuver per segment, returns from 0 - n segments
    choice = rng.choice(len(maneuvers), size=(base_S, n_segments), p=weights / weights.sum())

    # remove left/right bias
    sign = rng.choice([-1.0, 1.0], size=(base_S, n_segments))
    deltas = sign * rng.vonmises(mu=mus[choice], kappa=kappas[choice])

    # add to angles
    angles = np.empty((base_S, n_points))
    angles[:, 0]  = rng.uniform(-np.pi, np.pi, size=base_S)  # initial heading is free
    angles[:, 1:] = angles[:, 0:1] + np.cumsum(deltas, axis=1)      # deltas is -1 than angles

    # truncate to requested S now that all rng draws are done. trials 0..S-1 are identical
    # to whatever a larger-S caller would have gotten from the same seed.
    lengths = lengths[:S]
    times   = times[:S]
    angles  = angles[:S]

    # normalize --------------------------------------------------------------
    # motion fills n_segments / n_points of the timeline (origin->wp1 is non-moving)
    lengths = length * lengths                         # (S, segments + 1)
    times   = ((n_segments - 1) / n_points) * limit * times  # (S, segments)

    # gen segment pos boundaries, n_segments -> n_points boundaries
    paths     = lengths * np.exp(1j * angles)
    waypoints = np.cumsum(paths, axis=1)

    # gen segment time boundaries, n_segments -> n_points boundaries
    cumtime        = np.empty((S, n_points))
    cumtime[:, 0]  = 0                         # first column of every row
    cumtime[:, 1:] = np.cumsum(times, axis=1)  # remaining columns

    # generate timestamp for every tick in sim
    n_ticks = int(np.ceil(limit/dt))
    t       = np.arange(n_ticks) * dt

    # match every timestamps correspoinding segment, length = t range = 0..n_segments
    segment = np.empty((S, n_ticks), dtype=int) # shape (S, ticks)
    for s in range(S):
        segment[s, :] = np.searchsorted(cumtime[s], t, side='right') # gets index where sorted t would be inserted in cumtime for each S
    segment = np.clip(segment - 1, 0, n_segments - 1)                # insertion idx - 1 = segment idx, clip to exclude end boundary

    # get completion percentage, (timestamp - segment start time) / segment time length for each timestamp
    # super weird indexing here, basically segments is an array of column indexes pointing to cumtime column
    # each row of that is a different trial idx array, so you need the np.arrange at the start
    # so that each col pointer lines up with its respective segment
    # eg [0][1] means for tstamp 0 it pionts to segment 0 for first trial but to seg 1 in second trial etc
    # since theyre in order arrange lines them up to their respective trials
    trial_idx = np.arange(S, dtype=int)[:, np.newaxis] # for the extra dim
    fraction = (t - cumtime[trial_idx, segment]) / times[trial_idx, segment]  # type: ignore
    fraction = np.clip(fraction, 0, 1) # size (S, n_ticks)
    # min jerk easing: bell-shaped velocity per segment, zero accel at waypoints
    fraction = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)

    # calculate lerp position in segment
    # end point - start point * lerp frac + starting point
    t_pos = (waypoints[trial_idx, segment+1] - waypoints[trial_idx, segment]) * fraction + waypoints[trial_idx, segment] # type: ignore

    return t_pos


# individuals + sim settings -> simulation stats
def sim(individuals: list[Individual], settings, seed=None, ticks_per_drone: int = 0, log_per_tick: bool = False) -> tuple[list[Individual], dict]:
    # get configuration
    drone_conf = get_drone_conf(config_path)
    N = len(individuals) # drones
    with open(config_path, 'rb') as f:
        _cfg = tomllib.load(f)
        S          = _cfg['trainer']['trials']      # trials per drone
        input_dim  = _cfg['network']['layers'][0]   # network observation width
        action_dim = _cfg['network']['layers'][-1]  # network action width
    dt = .016

    # replay buffer preallocation. tuple = (s, a, r, s'). last tick excluded from
    # sampling so s' is always valid. ticks_per_drone draws per (drone, seed) episode.
    tuple_width = 2 * input_dim + action_dim + 1
    T_max = int(np.ceil(settings['limit'] / dt))
    if ticks_per_drone > 0:
        t_idx = np.random.randint(0, T_max - 1, size=(N, S, ticks_per_drone))   # [0, T_max-1)
        buffer = np.zeros((N, S, ticks_per_drone, tuple_width), dtype=np.float32)
    else:
        t_idx  = None
        buffer = None
    weight_penalty_coef = 1e-5   # tiny L2 penalty on genome weights, nudges toward simpler controllers

    # per-tick logging buffers (diagnostic only). when enabled, each tick's
    # raw increments (dt * track, dt * effort, dt * prox*pretouch, dt * score)
    # are stored so downstream tools can plot signal traces.
    if log_per_tick:
        T_alloc = int(np.ceil(settings['limit'] / dt))
        tick_fit        = np.zeros((N, S, T_alloc), dtype=np.float32)
        tick_track      = np.zeros((N, S, T_alloc), dtype=np.float32)
        tick_effort     = np.zeros((N, S, T_alloc), dtype=np.float32)
        tick_scale      = np.zeros((N, S, T_alloc), dtype=np.float32)
        tick_track_raw       = np.zeros((N, S, T_alloc), dtype=np.float32)  # pre-EMA
        tick_effort_raw      = np.zeros((N, S, T_alloc), dtype=np.float32)  # pre-criticality, pre-EMA
        tick_effort_weighted = np.zeros((N, S, T_alloc), dtype=np.float32)  # post-criticality, pre-EMA
        tick_discount        = np.zeros((N, S, T_alloc), dtype=np.float32)  # time-pressure decay, gamma**ticks_since_touch
        # drone position in the target's frame (drone - target, complex). |tick_rel|
        # gives distance-to-target; the complex value itself is the target-centered
        # trajectory. one buffer serves both the distance and trajectory diagnostics.
        tick_rel             = np.zeros((N, S, T_alloc), dtype=np.complex64)

    # initialize targets
    rng = np.random.default_rng(seed)
    # get the target position every tick
    tick_pos: np.ndarray = gen_target_chain(settings['length'], settings['limit'], dt, rng, S, base_S=S).astype(np.complex64)

    # init brain
    Brain = brain(individuals)

    # have physics states of all drones in one matrix
    # rows are drones, columns is state values.
    # perturbations flag (settings) gates init randomization + wind gusts. when False,
    # drones spawn at origin with zero velocity/angle/spin (clean state) and no gusts.
    perturbations = settings.get('perturbations', False)

    state_matrix = np.zeros((N, 7, S), dtype=np.complex64)
    state_matrix[:, 0, :] = 0j         # spawn at origin
    # pre-spool throttles to hover so random policies start in a viable regime.
    # neutral op point for rate-command thrust is gravity-matching, not zero.
    hover = drone_conf['M'] * abs(drone_conf['G']) / (2 * drone_conf['th_force'])
    state_matrix[:, 6, :] = hover + 1j * hover
    # columns 4, 5 (thruster angles) stay 0

    if perturbations:
        # randomized init, identical across drones on this seed
        angle0   = rng.uniform(-np.deg2rad(60), np.deg2rad(60))
        ang_vel0 = rng.uniform(-2.0, 2.0)

        v_init_max = 3
        v_mag = np.sqrt(rng.uniform(0, 1)) * v_init_max   # sqrt -> uniform 2D disk
        v_dir = rng.uniform(-np.pi, np.pi)
        vel0  = v_mag * np.exp(1j * v_dir)

        state_matrix[:, 1, :] = vel0
        state_matrix[:, 2, :] = angle0
        state_matrix[:, 3, :] = ang_vel0

    # impulse (wind gust) params — per-tick prob set so ~4 kicks happen per episode on average
    impulse_prob    = 4 * dt / settings['limit']
    impulse_v_sigma = 0.5
    impulse_w_sigma = 0.5

    # initial action matrix
    action_matrix = np.zeros((N, 4, S), dtype=np.float32)

    # simulation progression arrays
    toggle  = np.zeros((N, S), dtype=bool) # tracks initial touch to start chain
    ticks   = np.zeros((N, S), dtype=int) # ticks since touched
    crashed = np.zeros((N, S), dtype=bool) # latched flag: drone has flown out of bounds; zeros all future fitness
    crash_dist = 1.5 * settings['length']  # crash threshold: 1.5x chain length away from target
    # first-crash tick per (drone, seed). T_max sentinel = never crashed. used to drop
    # post-crash tuples from the replay buffer while keeping pre-crash ones.
    crash_tick = np.full((N, S), T_max, dtype=np.int32) if ticks_per_drone > 0 else None

    # TIME PRESSURE: ticks since the drone was last INSIDE the target (resets to 0 on
    # every touch, instantaneous — not the latched `toggle`). discount = touch_gamma **
    # ticks_since_touch, halving every 1/5 of the episode. applied to the PROXIMITY
    # component only (not bridge): the static-closeness floor decays but flying the
    # guidance law (track) keeps paying — kills the pre-touch hover-near-target farm.
    ticks_since_touch = np.zeros((N, S), dtype=np.int64)
    halflife_ticks    = max(1.0, np.ceil(settings['limit'] / dt) / 5.0)
    touch_gamma       = float(0.5 ** (1.0 / halflife_ticks))

    # fitness + descriptor arrays
    # descriptors: mean gimbal angle, activation variance
    fitness_velo  = np.zeros((N, S), dtype=np.float32)
    track_velo    = np.zeros((N, S), dtype=np.float32)  # diagnostic: ema track sum, sum(dt * track)
    effort_velo   = np.zeros((N, S), dtype=np.float32)  # diagnostic: raw effort sum, sum(dt * effort)
    scale_velo    = np.zeros((N, S), dtype=np.float32)  # diagnostic: scaling sum, sum(dt * prox)
    sum_acti2    = np.zeros((N, 4, S), dtype=np.float32)
    sum_acti     = np.zeros((N, 4, S), dtype=np.float32)
    mean_gimb    = np.zeros(N,         dtype=np.float32)
    total_ticks  = 0

    max_a = 2 * drone_conf['th_force'] / drone_conf['M']
    eps   = 1e-8
    eps_d = 0.05
    floor = 0.5    # hover tolerance for the tracking scale, m/s
    effort_floor = 0.05 * max_a * dt  # 5% of the dv budget; min divisor so near-zero err_v doesn't demand impossible precision

    # nested-headroom score knobs (both in [0,1]; keep <1 for the [0,1] bound + no-veto).
    LAMBDA_BRIDGE = 0.95  # fraction of the (1-prox) distance headroom tracking may fill
    MU_BRIDGE     = 0.30  # fraction of the (1-track) tracking headroom effort may fill

    # scale shape: f(x) = 1/(K·(x+A)) − A   where x = dist/L (normalized).
    # constants chosen so f(0)=1, f(1)=0, f(0.2)=0.2 — packs the dynamic range
    # into the hover regime (first 20% of L) so trained drones don't saturate at 1.
    SCALE_A = 1.0 / 15.0     # ≈ 0.0667
    SCALE_K = 225.0 / 16.0   # = 14.0625
    inv_L   = 1.0 / settings['length']

    # prenitialize values
    prev_los   = None
    prev_vel   = None
    # single-pole EMAs over the score components, both seeded at 1.0 so a startup transient
    # doesn't punish a smooth controller. effort uses a long K=16 window; tracking a tight
    # K=4 window — velocity error is noisy and needs smoothing, but a short horizon keeps it
    # responsive to a fast-moving target.
    K_effort     = 16
    alpha_effort = 1.0 / K_effort   # = 0.0625
    K_track      = 4
    alpha_track  = 1.0 / K_track    # = 0.25
    ema_effort   = np.ones((N, S), dtype=np.float32)
    ema_track    = np.ones((N, S), dtype=np.float32)
    time = 0

    # hoisted loop constants
    arangeS    = np.arange(S)
    norm_const = np.array([1, 1, 2, 2], dtype=np.float32).reshape(1, 4, 1)
    while time < settings['limit']:
        # alive mask: a drone is "alive" if at least one of its S seeds hasn't crashed.
        # fully-dead drones are skipped from physics + brain to save compute.
        alive_drones = ~crashed.all(axis=1)              # (N,) bool
        if not alive_drones.any():
            break                                         # whole population crashed, stop sim
        all_alive = bool(alive_drones.all())

        # UPDATE PHYSICS — full batch when nothing's dead, otherwise on the alive subset.
        if all_alive:
            state_matrix = physics_update(dt, state_matrix, action_matrix, drone_conf)
        else:
            sub_state = physics_update(dt, state_matrix[alive_drones], action_matrix[alive_drones], drone_conf)
            state_matrix[alive_drones] = sub_state

        # RANDOM IMPULSES (wind gusts) — shared across drones within a seed for fairness.
        # gated by perturbations flag; disabled gives clean trajectories with no external kicks.
        if perturbations:
            mask   = rng.random(S) < impulse_prob
            v_kick = (rng.normal(0, impulse_v_sigma, S) + 1j * rng.normal(0, impulse_v_sigma, S)) * mask
            w_kick = rng.normal(0, impulse_w_sigma, S) * mask
            state_matrix[:, 1, :] += v_kick
            state_matrix[:, 3, :] += w_kick

        # MAKE OBSERVATION ARRAY -----------------------------------------------------------
        # obs values (27)  -- each vector as [magnitude, unit_x, unit_y]; frame in []
        #   target geometry:
        #     1. dist            2. los ux     3. los uy      [local]
        #     4. los rate [world]  5. tti [scalar]
        #   self velocity [local]:        6. |vel|     7. ux   8. uy
        #   relative velocity [local]:    9. |relvel| 10. ux  11. uy
        #   prev-tick net accel [local]: 12. |acc|    13. ux  14. uy
        #   guidance accel cmds [local] (mag tanh-capped to max_a):
        #    15. |zem_a| 16. ux 17. uy        18. |zev_a| 19. ux 20. uy
        #   attitude [world]: 21. sin(angle) 22. cos(angle) 23. ang vel
        #   actuators [local/cmd]: 24. t1 angle 25. t2 angle 26. t1 thrust 27. t2 thrust (last tick)
        angle        = state_matrix[:, 2, :].real # type:ignore
        target       = tick_pos[arangeS, ticks]
        delta_world  = target - state_matrix[:, 0, :]
        delta_local  = delta_world * np.exp(-1j * angle)

        # --- target geometry ---
        dist  = np.abs(delta_world)                # range (magnitude)
        los_u = delta_world / (dist + eps)         # world unit dir (for rate)
        los_local = delta_local / (dist + eps)     # local unit dir
        if prev_los is None:
            prev_los = los_u
        # los rate in LOCAL frame = inertial los rate - drone angular velocity
        # (los_u stays world for the calc; only the obs value is converted to body)
        los_rate = np.tanh((np.angle(los_u / prev_los) / dt - state_matrix[:, 3, :].real) / 3.0)
        prev_los = los_u

        # --- self velocity (local) ---
        vel = state_matrix[:, 1, :] * np.exp(-1j * angle)
        vel_mag = np.abs(vel)
        vel_u   = vel / (vel_mag + eps)

        # --- relative velocity (drone - target, local) ---
        prev_ticks  = np.maximum(ticks - 1, 0)
        v_target    = (target - tick_pos[arangeS, prev_ticks]) / dt
        v_target_local = v_target * np.exp(-1j * angle)
        rel_vel = vel - v_target_local
        relvel_mag = np.abs(rel_vel)
        relvel_u   = rel_vel / (relvel_mag + eps)

        # --- prev-tick net acceleration (world -> local) ---
        vel_world = state_matrix[:, 1, :].copy()   # copy: state_matrix is mutated in place next tick
        if prev_vel is None:
            prev_vel = vel_world
        net_acc = ((vel_world - prev_vel) / dt) * np.exp(-1j * angle)
        acc_mag = np.abs(net_acc)
        acc_u   = net_acc / (acc_mag + eps)

        # --- time to closest approach ---
        closest_approach = (delta_local * np.conjugate(rel_vel)).real / (np.abs(rel_vel) + eps)
        tti_raw = closest_approach / (np.abs(rel_vel) + eps)
        tti_obs = np.tanh(tti_raw / 10.0)

        # --- guidance accel commands: ZEM/ZEV at t_go = tti_raw w/ gravity -> PN form, mag-capped ---
        grav_body = (-1j * drone_conf['G']) * np.exp(-1j * angle)  # world gravity -> body frame
        zem = delta_local - (rel_vel * tti_raw + 0.5 * grav_body * tti_raw ** 2)
        zev = v_target_local - (vel + grav_body * tti_raw)
        zem_a = zem / (tti_raw ** 2 + eps)
        zev_a = zev / (tti_raw + eps)
        zem_a = zem_a / (np.abs(zem_a) + eps) * np.tanh(np.abs(zem_a) / max_a)
        zev_a = zev_a / (np.abs(zev_a) + eps) * np.tanh(np.abs(zev_a) / max_a)
        zema_mag = np.abs(zem_a); zema_u = zem_a / (zema_mag + eps)
        zeva_mag = np.abs(zev_a); zeva_u = zev_a / (zeva_mag + eps)

        # --- attitude / actuators ---
        ang_vel   = state_matrix[:, 3, :].real # type:ignore
        t1_ang    = state_matrix[:, 4, :].real # type:ignore
        t2_ang    = state_matrix[:, 5, :].real # type:ignore
        # actual integrated thrust level (not the command) — what the drone is actually feeling.
        # left in [0, 1] to match the sigmoid output scale on the same channels.
        t1_thrust = state_matrix[:, 6, :].real
        t2_thrust = state_matrix[:, 6, :].imag

        obs = np.stack([
            dist / 10.0, los_local.real, los_local.imag, los_rate, tti_obs,
            vel_mag / 10.0, vel_u.real, vel_u.imag,
            relvel_mag / 10.0, relvel_u.real, relvel_u.imag,
            acc_mag, acc_u.real, acc_u.imag,
            zema_mag, zema_u.real, zema_u.imag,
            zeva_mag, zeva_u.real, zeva_u.imag,
            np.sin(angle), np.cos(angle), ang_vel,
            t1_ang, t2_ang, t1_thrust, t2_thrust,
        ], axis=1) # have to have 3 dim for forward pass (N, inputs, S)

        # FORWARD PASS OBSERVATIONS --------------------------------------------------------
        # pass alive mask so brain skips fully-dead drones; their outputs hold last tick's.
        if all_alive:
            action_matrix = Brain.forward(obs)
        else:
            action_matrix = Brain.forward(obs, alive=alive_drones, prev_actions=action_matrix)

        # PROGRESS SIMULATION --------------------------------------------------------------
        # check if touched waypoint
        touching = dist < 0.5
        toggle |= touching

        # time-pressure counter: reset on touch, else increment. computed BEFORE the score
        # so a touch tick gets full credit (discount = touch_gamma ** 0 = 1).
        ticks_since_touch = np.where(touching, 0, ticks_since_touch + 1)
        discount          = touch_gamma ** ticks_since_touch

        # crash detection (latched): drone has flown more than crash_dist away from target.
        # zero crash tolerance: any (drone, seed) pair that ever crashes gets its FULL
        # episode total zeroed at the end of the sim — not just post-crash ticks.
        prev_crashed = crashed.copy() if crash_tick is not None else None
        crashed |= dist > crash_dist
        # latch the iteration of each pair's first crash for buffer post-filtering.
        if crash_tick is not None:
            newly_crashed = crashed & ~prev_crashed
            crash_tick = np.where(newly_crashed, total_ticks, crash_tick)

        # FITNESS CALULATIONS --------------------------------------------------------------
        # SHARED: ideal drone vel VECTOR = match target motion + approach budget along los_u.
        # approach budget is purely along los_u (toward target); perpendicular ideal is just
        # the target's lateral drift, so the vector form scores it for free. used by both
        # the tracking and effort components below.
        v_tgt_par    = (v_target * np.conj(los_u)).real      # target vel along approach dir (N, S)
        safe_v       = 0.8 * np.sqrt(2 * max_a * (dist + eps_d))   # max safe approach velocity (N, S)
        # smoothly taper the approach budget to 0 inside the touch radius. linear scale turns
        # the sqrt profile into dist^1.5 near the target (zero slope at origin) -> no velocity-
        # command cliff at 0.5 and no overshoot-inducing steep sqrt tangent. collapses to hover.
        smooth_scale = np.clip(dist / 0.5, 0.0, 1.0)
        safe_v_term  = safe_v * smooth_scale
        ideal_vel    = v_target + safe_v_term * los_u

        # TRACKING component (vratio, error form, hover-safe) ------------------------------
        # how close the achieved velocity actually is to ideal_vel. anchors the reward to the
        # high-level goal so the drone cant farm effort points from a self-made bad state.
        track_err   = np.abs(state_matrix[:, 1, :] - ideal_vel)
        track_scale = np.maximum(np.abs(ideal_vel), floor)  # floored for hover
        track_raw   = np.clip(1.0 - (track_err / track_scale) ** 2, 0.0, 1.0)

        # single-pole EMA on tracking (K=4) — velocity error is noisy; a tight window
        # smooths chatter without lagging a fast target.
        ema_track = alpha_track * track_raw + (1.0 - alpha_track) * ema_track
        track = ema_track

        # EFFORT component (regime-aware projection match) ----------------------------------
        # ideal_projection = how much actual_dv SHOULD project along err_unit this tick
        # (capped at dv budget). divisor floors at effort_floor so near-equilibrium ticks
        # dont demand impossible precision. penalizes both over- and under-shoot equally.
        v_free = prev_vel + -9.81j * dt
        # per-tick impulse capacity = max_a * dt (full-thrust velocity change in one tick).
        # NOT scaled by actuation_rate*dt: that's the throttle slew-rate (how fast thrust
        # can CHANGE), not the impulse a tick delivers. once at a given throttle every tick
        # delivers its full impulse, so the slew factor under-budgeted the ideal ~6x.
        dv = max_a * dt

        err_v    = ideal_vel - v_free
        err_mag  = np.abs(err_v)
        err_unit = err_v / (err_mag + eps)

        actual_dv        = state_matrix[:, 1, :] - v_free
        ideal_projection = np.minimum(err_mag, dv)          # capped ideal magnitude along err
        ideal_effort     = err_unit * ideal_projection      # full 2-D ideal effort vector (û · p*)
        effort_divisor   = np.maximum(ideal_projection, effort_floor)

        prev_vel   = vel_world
        # 2-D miss between the actual thrust impulse and the capped ideal vector. only
        # differs from the scalar projection when |err| < dv (near/hover): there the drone
        # has spare lateral budget it could waste, and this charges for that perpendicular
        # thrust. when |err| >= dv the whole budget is along err, so it reduces to scalar.
        effort_err = np.abs(actual_dv - ideal_effort)
        effort_raw = np.clip(1.0 - (effort_err / effort_divisor) ** 2, 0.0, 1.0)

        # single-pole EMA over the RAW effort score (criticality gate removed — the
        # nested-headroom (1-track) factor below now gates when effort matters).
        ema_effort = alpha_effort * effort_raw + (1.0 - alpha_effort) * ema_effort
        effort = ema_effort

        # FINAL score = nested-headroom priority cascade: distance > tracking > effort.
        # prox is the floor; tracking may spend only the (1-prox) distance headroom;
        # effort may spend only the (1-track) sub-headroom inside that. closeness is
        # never vetoed (d_score/d_prox = 1 - LAMBDA*bridge >= 1 - LAMBDA > 0). per-tick
        # score stays in [0,1], = 1 only at the target (prox=1). ------------------------
        x_norm = dist * inv_L
        prox = np.maximum(0.0, 1.0 / (SCALE_K * (x_norm + SCALE_A)) - SCALE_A)

        bridge = track + MU_BRIDGE * (1.0 - track) * effort
        # time-pressure discount applied to PROXIMITY ONLY. bridge fills the (1 - prox_eff)
        # headroom so the cascade stays in [0,1] and bridge gains headroom as prox decays:
        # the static closeness floor bleeds away, but tracking the guidance law (which
        # points at the target whenever you're outside it) remains fully paid. removes the
        # pre-touch hover-near-target exploit without vetoing legitimate proximity.
        prox_eff = prox * discount
        score    = prox_eff + LAMBDA_BRIDGE * (1.0 - prox_eff) * bridge
        fitness_velo += dt * score                          # (N, S) final fitness
        track_velo   += dt * track                          # ema-smoothed track quality (undiscounted)
        effort_velo  += dt * effort                         # ema-smoothed effort quality (undiscounted)
        scale_velo   += dt * prox                           # proximity scaling (undiscounted)

        # REPLAY BUFFER WRITES -------------------------------------------------------------
        # tuple at sampled tick t = (s_t, a_t, r_t, s_{t+1}).
        # at this iteration (total_ticks): obs = s_t, action_matrix = a_t, dt*score = r_{t-1}.
        # so:  match t_idx == total_ticks       -> write s, a
        #      match t_idx == total_ticks - 1   -> write r (this iter's score) and s' (this iter's obs)
        if t_idx is not None:
            match_sa = (t_idx == total_ticks)            # (N, S, k)
            if match_sa.any():
                n_i, s_i, k_i = np.where(match_sa)
                buffer[n_i, s_i, k_i, :input_dim] = obs[n_i, :, s_i]
                buffer[n_i, s_i, k_i, input_dim:input_dim + action_dim] = action_matrix[n_i, :, s_i]
            if total_ticks > 0:
                match_rs = (t_idx == total_ticks - 1)
                if match_rs.any():
                    n_i, s_i, k_i = np.where(match_rs)
                    buffer[n_i, s_i, k_i, input_dim + action_dim] = dt * score[n_i, s_i]
                    buffer[n_i, s_i, k_i, input_dim + action_dim + 1:] = obs[n_i, :, s_i]

        if log_per_tick:
            tick_fit            [:, :, total_ticks] = dt * score
            tick_track          [:, :, total_ticks] = dt * track             # ema-smoothed
            tick_effort         [:, :, total_ticks] = dt * effort            # ema-smoothed
            tick_scale          [:, :, total_ticks] = dt * prox
            tick_track_raw      [:, :, total_ticks] = dt * track_raw
            tick_effort_raw     [:, :, total_ticks] = dt * effort_raw        # raw effort, pre-EMA
            tick_effort_weighted[:, :, total_ticks] = dt * effort_raw        # criticality removed; == raw
            tick_discount       [:, :, total_ticks] = discount               # time-pressure decay (unscaled)
            tick_rel            [:, :, total_ticks] = -delta_world           # drone pos in target frame

        # DESCRIPTOR CALCULATIONS ----------------------------------------------------------
        # update descriptors
        normalized = action_matrix / norm_const
        sum_acti += normalized       # (N, 4, S)
        sum_acti2 += normalized ** 2 # (N, 4, S)
        mean_gimb += ((np.abs(state_matrix[:, 4, :]) + np.abs(state_matrix[:, 5, :])) / 2).mean(axis=1) # (N, )

        # forward ticks it toggled
        ticks += toggle
        total_ticks += 1

        time += dt

    # zero crash tolerance: any (drone, seed) pair that ever crashed gets its full
    # episode contribution wiped — fitness, track, effort, scale all set to 0.
    survived = (~crashed).astype(np.float32)
    fitness_velo *= survived
    track_velo   *= survived
    effort_velo  *= survived
    scale_velo   *= survived

    # replay buffer: drop only the tuples whose sampled tick was at or after the
    # (drone, seed)'s crash. pre-crash ticks from a later-crashing pair are kept.
    if buffer is not None and crash_tick is not None and t_idx is not None:
        valid  = t_idx < crash_tick[:, :, None]   # (N, S, ticks_per_drone)
        buffer = buffer[valid]                    # (n_valid, tuple_width)

    # finalize descriptors and fitness
    var_acti = (sum_acti2 / total_ticks) - (sum_acti / total_ticks)**2 # (N, act, S)
    var_acti = var_acti.mean(axis=(1, 2)) # (N, )
    mean_gimb = mean_gimb / total_ticks   # (N, )

    # floor whole-episode fitness at 0, negatives within episode still shape gradient
    # fitness = np.maximum(fitness_velo, 0.0)
    fitness = fitness_velo

    # per-drone fitness: SE-penalized mean (mean - std/sqrt(S)).
    # this is the value that drives selection everywhere — archive ranking, bandit
    # scoring, CMA parent picks, parallel_sim's top-K. drones with consistent good
    # performance across seeds beat lucky-spike drones. raw per-drone signals
    # (per_track / per_effort / per_scale / tick_* / fit_mean) stay raw so the
    # diagnostic plots and trend metrics read uncontaminated.
    per_seed_mean = fitness.mean(axis=1)                 # raw per-drone mean (N,)
    per_seed_std  = fitness.std (axis=1)                 # raw per-drone std  (N,)
    per_drone_fit = per_seed_mean - per_seed_std / np.sqrt(S)

    # tiny parsimony pressure: penalize mean-squared genome weight magnitude so that,
    # all else equal, smaller/simpler weight vectors win out (smoother, less overfit
    # controllers). mean (not sum) keeps this independent of genome size; coefficient
    # is small enough to act only as a tie-breaker, not to outweigh the fitness signal.
    weight_l2 = np.array([np.mean(ind.weights**2) for ind in individuals])
    per_drone_fit = per_drone_fit - weight_penalty_coef * weight_l2

    top_idx = per_drone_fit.argsort()[-5:]
    top = fitness[top_idx, :]

    # print(f'in drone std {top.std(axis=1).mean(): .2f}, total std {top.std(): .2f}, cross drone std {top.mean(axis=1).std(): .2f}, arm {individuals[top_idx[-1]].tag}')

    for i, ind in enumerate(individuals):
        ind.fitness = float(per_drone_fit[i])
        ind.descriptors = {'mean_gimb': mean_gimb[i], 'var_action': var_acti[i]}

    stats_out = {'fit_mean'   : fitness.mean(),                      # raw population mean
                 'fit_max'    : float(per_drone_fit.max()),          # selection-relevant (SE'd)
                 'fit_velo'   : fitness_velo.mean(),
                 # per-drone arrays. per_fit is SE form (drives parallel_sim top-K).
                 # per_fit_raw, per_fit_std, per_track/effort/scale stay raw for diagnostics.
                 'per_fit'    : per_drone_fit,
                 'per_fit_raw': per_seed_mean,
                 'per_fit_std': per_seed_std,
                 'per_track'  : track_velo.mean(axis=1),
                 'per_effort' : effort_velo.mean(axis=1),
                 'per_scale'  : scale_velo.mean(axis=1),
                 'S'          : S,
                 'buffer'     : buffer.reshape(-1, tuple_width) if buffer is not None else None}

    if log_per_tick:
        # trim to actual ticks executed (in case loop exited early via all-crashed break)
        stats_out['tick_fit']        = tick_fit        [:, :, :total_ticks]
        stats_out['tick_track']           = tick_track          [:, :, :total_ticks]
        stats_out['tick_effort']          = tick_effort         [:, :, :total_ticks]
        stats_out['tick_scale']           = tick_scale          [:, :, :total_ticks]
        stats_out['tick_track_raw']       = tick_track_raw      [:, :, :total_ticks]
        stats_out['tick_effort_raw']      = tick_effort_raw     [:, :, :total_ticks]
        stats_out['tick_effort_weighted'] = tick_effort_weighted[:, :, :total_ticks]
        stats_out['tick_discount']        = tick_discount       [:, :, :total_ticks]
        stats_out['tick_rel']             = tick_rel            [:, :, :total_ticks]
        stats_out['dt']                   = dt

    return individuals, stats_out

def parallel_sim(indivs: list[Individual], settings, Mpool: Pool, seed=None) -> tuple[list[Individual], dict]:
    # get ocpu count
    cpus = os.cpu_count()
    cpus = 0 if cpus is None else cpus
    if cpus == 0:
        raise RuntimeError('os.cpu_count() returned None')

    # replay buffer sizing — per (drone, seed) sample count.
    # ticks_per_drone = ceil(tuples_per_gen / (pop_size * trials)).
    with open(config_path, 'rb') as f:
        cfg = tomllib.load(f)
    tuples_per_gen = cfg['critic']['tuples_per_gen']
    S              = cfg['trainer']['trials']
    pop            = len(indivs)
    ticks_per_drone = -(-tuples_per_gen // (pop * S))  # ceil div

    # create the chunks
    chunk_size = len(indivs) // max(cpus, 1)
    chunks = []
    for i in range(cpus):
        start = i * chunk_size
        if i+1 == cpus:
            chunk = indivs[start: ]
        else:
            end   = (i+1) * chunk_size
            chunk = indivs[start: end]
        chunks.append(chunk)

    args = [(chunk, settings, seed, ticks_per_drone) for chunk in chunks]
    chunk_results = Mpool.starmap(sim, args)

    scored = []
    stats  = {'fit_max': -np.inf, 'fit_mean': 0.0, 'fit_velo': 0.0,
              'fit_track': 0.0, 'fit_effort': 0.0, 'fit_scale': 0.0}
    per_fit_all, per_fit_raw_all = [], []
    per_track_all, per_effort_all, per_scale_all = [], [], []
    per_fit_std_all = []
    buffer_chunks = []
    S_val = None
    for indvs, stat in chunk_results:
        scored.extend(indvs)
        stats['fit_mean']   += stat['fit_mean']
        stats['fit_velo']   += stat['fit_velo']
        stats['fit_max' ] =  max(stats['fit_max' ], stat['fit_max' ])
        per_fit_all    .append(stat['per_fit'])
        per_fit_raw_all.append(stat['per_fit_raw'])
        per_track_all  .append(stat['per_track'])
        per_effort_all .append(stat['per_effort'])
        per_scale_all  .append(stat['per_scale'])
        per_fit_std_all.append(stat['per_fit_std'])
        if stat['buffer'] is not None:
            buffer_chunks.append(stat['buffer'])
        S_val = stat['S']

    # compile replay buffers across chunks. axis 0 = episodes ((drone, seed) pairs).
    # stats['buffer'] = np.concatenate(buffer_chunks, axis=0) if buffer_chunks else None
    # if stats['buffer'] is not None:
    #     buf = stats['buffer']
    #     with open(config_path, 'rb') as f:
    #         _c = tomllib.load(f)
    #         in_d  = _c['network']['layers'][0]
    #         act_d = _c['network']['layers'][-1]
    #     np.set_printoptions(precision=3, suppress=True, linewidth=200)
    #     print(f"\n--- buffer test log: shape {buf.shape} ---")
    #     rows = np.random.choice(buf.shape[0], size=min(3, buf.shape[0]), replace=False)
    #     for r in rows:
    #         s   = buf[r, :in_d]
    #         a   = buf[r, in_d:in_d + act_d]
    #         rew = buf[r, in_d + act_d]
    #         sn  = buf[r, in_d + act_d + 1:]
    #         print(f"  row {r}:")
    #         print(f"    s_t        = {s}")
    #         print(f"    a_t        = {a}")
    #         print(f"    r_t        = {rew:.6f}")
    #         print(f"    s_{{t+1}}    = {sn}")
    #         print(f"    ds = s'-s  = {sn - s}")
    #     all_zero = (buf == 0).all(axis=1).sum()
    #     print(f"  fully-zero rows: {all_zero}/{buf.shape[0]}  (should be 0 if every slot was written)")

    stats['fit_mean']   /= cpus
    stats['fit_velo']   /= cpus

    # global top 10 by per-drone SE-fitness (matches selection); report their raw component means
    per_fit     = np.concatenate(per_fit_all)        # SE form
    per_fit_raw = np.concatenate(per_fit_raw_all)    # raw mean
    per_track   = np.concatenate(per_track_all)
    per_effort  = np.concatenate(per_effort_all)
    per_scale   = np.concatenate(per_scale_all)
    k = min(10, per_fit.size)
    top10 = np.argpartition(per_fit, -k)[-k:]
    stats['fit_track']  = float(per_track [top10].mean())
    stats['fit_effort'] = float(per_effort[top10].mean())
    stats['fit_scale']  = float(per_scale  [top10].mean())

    print(f"track, {stats['fit_track']:.2f}, effort, {stats['fit_effort']:.2f}, scale, {stats['fit_scale']:.2f}")

    # SEM diagnostic is on RAW values so the noise ratio is comparable across versions
    # (sem comes from the raw per-drone std; pop_mean/cross_std use raw per-drone means).
    per_fit_std = np.concatenate(per_fit_std_all)
    sem        = (per_fit_std / np.sqrt(S_val)).mean()       # mean across drones of per-drone SEM
    pop_mean   = per_fit_raw.mean()                          # raw mean fitness across drones
    cross_std  = per_fit_raw.std()                           # raw std across drones (of seed-mean)
    ratio_mean = sem / pop_mean if pop_mean != 0 else float('inf')
    ratio_std  = sem / cross_std if cross_std != 0 else float('inf')
    print(f"SEM/pop_mean, {ratio_mean:.3f}, SEM/cross_std, {ratio_std:.3f}, S, {S_val}")

    return scored, stats


if __name__=="__main__":
    import time
    from modules.evo_alg.stub import evostub

    alg = evostub()
    qty = 1000
    indv, _ = alg.propose(qty, None)

    limit = 1
    t0 = time.perf_counter()
    sim(indv, {'limit': limit, 'length': 10})
    sim_time = time.perf_counter() - t0
    print(f"pop size: {qty}")
    print(f"sim time: {sim_time:.3f}s")
    print(f"per drone: {sim_time*1000/qty:.3f}ms")
    print(f"realtime factor: {limit/sim_time:.3f}x")
