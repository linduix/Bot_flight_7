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
        'th_force'          : drone_config['thruster_force']
    }
    return drone

# physics state
# 0. x + i*y
# 1. vx + i*vx
# 2. angle
# 3. angular vel
# 4. t1 angle
# 5. t2 angle
state = [0+0j, 0+0j, 0, 0, 0, 0]

# physics state matrix + actions matrix -> updated state
def physics_update(dt, state: np.ndarray, actions: np.ndarray, drone_conf: dict):
    assert state.ndim   == 3, f"state must be 3D (n, state, S), got shape {state.shape}"
    assert actions.ndim == 3, f"actions must be 3D (n, action, S), got shape {actions.shape}"

    # process actions
    t1, t2, rot1, rot2 = actions[:, 0, :], actions[:, 1, :], actions[:, 2, :], actions[:, 3, :]
    t1, t2 = np.maximum(0, t1), np.maximum(0, t2)

    # calculate forces (drone refrence frame) --------------------
    rotation_speed = float(np.deg2rad(drone_conf['th_rotation_speed']))
    # thruster rotation
    state[:, 4, :] += rotation_speed * rot1 * dt
    state[:, 5, :] += rotation_speed * rot2 * dt
    # ADD ROTATION CLIPPING
    max_angle = np.deg2rad(drone_conf['th_max_angle'])
    state[:, 4, :] = np.clip(state[:, 4, :].real, -max_angle, max_angle) # type:ignore
    state[:, 5, :] = np.clip(state[:, 5, :].real, -max_angle, max_angle) # type:ignore

    # THRUST
    # magnitude
    thrust1 = t1 * drone_conf['th_force']
    thrust2 = t2 * drone_conf['th_force']

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

def gen_target_chain(length, limit, dt, rng, S) -> np.ndarray:
    # purpose: to map every tick in sim to the position of target after touch
    n_segments = 5
    n_points   = n_segments + 1   # origin -> wp1 + motion segments

    # path length + speed -------------
    alpha = 3.0
    concentration = 15
    lengths = rng.dirichlet([alpha] * n_points, size=S)          # including origin -> wp1 (S, segments + 1)
    # have to do manual dirclet for times cuz no brodcasting for this func
    # times   = rng.dirichlet(lengths[1:] * concentration)         # NOT including origin -> wp1 (S, segments)
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
    choice = rng.choice(len(maneuvers), size=(S, n_segments), p=weights / weights.sum())

    # remove left/right bias
    sign = rng.choice([-1.0, 1.0], size=(S, n_segments))
    deltas = sign * rng.vonmises(mu=mus[choice], kappa=kappas[choice])

    # add to angles
    angles = np.empty((S, n_points))
    angles[:, 0]  = rng.uniform(-np.pi, np.pi, size=S)  # initial heading is free
    angles[:, 1:] = angles[:, 0:1] + np.cumsum(deltas, axis=1)      # deltas is -1 than angles

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
def sim(individuals: list[Individual], settings, seed=None) -> tuple[list[Individual], dict]:
    # get configuration
    drone_conf = get_drone_conf(config_path)
    N = len(individuals) # drones
    S = 16                # trials per drone
    dt = .016

    # initialize targets
    rng = np.random.default_rng(seed)
    # get the target position every tick
    tick_pos: np.ndarray = gen_target_chain(settings['length'], settings['limit'], dt, rng, S).astype(np.complex64)

    # init brain
    Brain = brain(individuals)

    # have physics states of all drones in one matrix
    # rows are drones, columns is state values
    # randomized init, identical across drones on this seed
    angle0   = rng.uniform(-np.deg2rad(60), np.deg2rad(60))
    ang_vel0 = rng.uniform(-2.0, 2.0)

    v_init_max = 3
    v_mag = np.sqrt(rng.uniform(0, 1)) * v_init_max   # sqrt -> uniform 2D disk
    v_dir = rng.uniform(-np.pi, np.pi)
    vel0  = v_mag * np.exp(1j * v_dir)

    state_matrix = np.zeros((N, 6, S), dtype=np.complex64)
    state_matrix[:, 0, :] = 0j         # spawn at origin
    state_matrix[:, 1, :] = vel0
    state_matrix[:, 2, :] = angle0
    state_matrix[:, 3, :] = ang_vel0
    # columns 4, 5 (thruster angles) stay 0

    # impulse (wind gust) params
    impulse_prob    = 0.01
    impulse_v_sigma = 1.5
    impulse_w_sigma = 1.5

    # initial action matrix
    action_matrix = np.zeros((N, 4, S), dtype=np.float32)

    # simulation progression arrays
    toggle  = np.zeros((N, S), dtype=bool) # tracks initial touch to start chain
    ticks   = np.zeros((N, S), dtype=int) # ticks since touched

    # fitness + descriptor arrays
    # descriptors: mean gimbal angle, activation variance
    fitness_velo = np.zeros((N, S),    dtype=np.float32)
    sum_acti2    = np.zeros((N, 4, S), dtype=np.float32)
    sum_acti     = np.zeros((N, 4, S), dtype=np.float32)
    mean_gimb    = np.zeros(N,         dtype=np.float32)
    total_ticks  = 0

    max_a = 2 * drone_conf['th_force'] / drone_conf['M']
    eps   = 1e-8
    eps_d = 0.05
    floor = 0.5   # hover tolerance, m/s
    
    # prenitialize values
    prev_los   = None
    prev_vel   = None
    time = 0

    # hoisted loop constants
    arangeS    = np.arange(S)
    norm_const = np.array([1, 1, 2, 2], dtype=np.float32).reshape(1, 4, 1)
    while time < settings['limit']:
        # UPDATE PHYSICS
        state_matrix = physics_update(dt, state_matrix, action_matrix, drone_conf)

        # RANDOM IMPULSES (wind gusts) — shared across drones within a seed for fairness
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
        vel_world = state_matrix[:, 1, :]
        if prev_vel is None:
            prev_vel = vel_world
        net_acc = ((vel_world - prev_vel) / dt) * np.exp(-1j * angle)
        prev_vel = vel_world
        acc_mag = np.abs(net_acc)
        acc_u   = net_acc / (acc_mag + eps)

        # --- time to closest approach ---
        closest_approach = (delta_local * np.conjugate(rel_vel)).real / (np.abs(rel_vel) + eps)
        tti_raw = closest_approach / (np.abs(rel_vel) + eps)
        tti_obs = np.tanh(tti_raw / 10.0)

        # --- guidance accel commands (zem/zev projected 1s w/ gravity -> PN form, mag-capped) ---
        horizon   = 1.0
        grav_body = (-1j * drone_conf['G']) * np.exp(-1j * angle)  # world gravity -> body frame
        zem = delta_local - rel_vel * horizon + 0.5 * grav_body * horizon ** 2
        zev = v_target_local - vel + grav_body * horizon
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
        t1_thrust = action_matrix[:, 0, :]     # last tick's thrust commands
        t2_thrust = action_matrix[:, 1, :]

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
        action_matrix = Brain.forward(obs)

        # PROGRESS SIMULATION --------------------------------------------------------------
        # check if touched waypoint
        toggle |= dist < 0.5

        # FITNESS CALULATIONS --------------------------------------------------------------
        # vratio fitness component (error form, hover-safe):
        # target vel projected onto approach direction (N, S)
        v_tgt_par = (v_target * np.conj(los_u)).real
        # max safe approach velocity (N, S)
        safe_v    = np.sqrt(2 * max_a * (dist + eps_d))
        # smoothly taper the approach budget to 0 inside the touch radius. linear
        # scale turns the sqrt profile into dist^1.5 near the target (zero slope at
        # origin) -> no velocity-command cliff at 0.5 and no overshoot-inducing
        # steep sqrt tangent right at the target. collapses to pure hover-match.
        smooth_scale = np.clip(dist / 0.5, 0.0, 1.0)
        safe_v_term  = safe_v * smooth_scale
        # ideal drone vel VECTOR = match target motion + approach budget along los_u.
        # approach budget is purely along los_u (toward target); perpendicular ideal is
        # just the target's lateral drift, so the vector form scores it for free.
        ideal_vel = v_target + safe_v_term * los_u

        # full 2D velocity error magnitude (N, S). expands to sqrt(par_err^2 +
        # perp_err^2): the parallel piece is identical to the old projection form,
        # the perpendicular piece penalizes orbiting / lateral drift beyond target.
        err   = np.abs(state_matrix[:, 1, :] - ideal_vel)
        # tolerance from the parallel ideal magnitude, floored so hover (ideal=0)
        # doesnt explode (N, S). large during fast approach -> lenient on lateral
        # velocity; floors at hover -> tight, so orbiting is punished where it hurts.
        ideal_par = v_tgt_par + safe_v_term
        scale = np.maximum(np.abs(ideal_par), floor)
        # inverted quadratic centered at err=0, mild negative for wrong-way ticks
        score = np.clip(1.0 - (err / scale) ** 2, -0.1, 1.0)
        score = score / (1 + np.sqrt(dist))

        fitness_velo += dt * score # (N, S)

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

    # finalize descriptors and fitness
    var_acti = (sum_acti2 / total_ticks) - (sum_acti / total_ticks)**2 # (N, act, S)
    var_acti = var_acti.mean(axis=(1, 2)) # (N, )
    mean_gimb = mean_gimb / total_ticks   # (N, )

    # floor whole-episode fitness at 0, negatives within episode still shape gradient
    # fitness = np.maximum(fitness_velo, 0.0)
    fitness = fitness_velo

    top_idx = fitness.mean(axis=1).argsort()[-5:]
    top = fitness[top_idx, :]

    # print(f'in drone std {top.std(axis=1).mean(): .2f}, total std {top.std(): .2f}, cross drone std {top.mean(axis=1).std(): .2f}, arm {individuals[top_idx[-1]].tag}')

    for i, ind in enumerate(individuals):
        ind.fitness = fitness[i, :].mean()
        ind.descriptors = {'mean_gimb': mean_gimb[i], 'var_action': var_acti[i]}

    return individuals, {'fit_mean': fitness.mean(), 'fit_max': fitness.mean(axis=1).max(),
                         'fit_velo': fitness_velo.mean()}

def parallel_sim(indivs: list[Individual], settings, Mpool: Pool, seed=None) -> tuple[list[Individual], dict]:
    # get ocpu count
    cpus = os.cpu_count()
    cpus = 0 if cpus is None else cpus
    if cpus == 0:
        raise RuntimeError('os.cpu_count() returned None')

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

    args = [(chunk, settings, seed) for chunk in chunks]
    chunk_results = Mpool.starmap(sim, args)

    scored = []
    stats  = {'fit_max': -np.inf, 'fit_mean': 0.0, 'fit_velo': 0.0}
    for indvs, stat in chunk_results:
        scored.extend(indvs)
        stats['fit_mean'] += stat['fit_mean']
        stats['fit_velo'] += stat['fit_velo']
        stats['fit_max' ] =  max(stats['fit_max' ], stat['fit_max' ])

    stats['fit_mean'] /= cpus
    stats['fit_velo'] /= cpus

    print(f"fitness, {stats['fit_velo']:.2f}, {stats['fit_mean']:.2f}")

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
