from modules.individual import Individual
from modules.batch_brain import brain
import numpy as np
import tomllib


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
    assert state.ndim   == 2, f"state must be 3D (n, state), got shape {state.shape}"
    assert actions.ndim == 2, f"actions must be 3D (n, action), got shape {actions.shape}"

    # process actions
    t1, t2, rot1, rot2 = actions[:, 0], actions[:, 1], actions[:, 2], actions[:, 3]
    t1, t2 = np.maximum(0, t1), np.maximum(0, t2)

    # calculate forces (drone refrence frame) --------------------
    rotation_speed = np.deg2rad(drone_conf['th_rotation_speed'])
    # thruster rotation
    state[:, 4] += rotation_speed * rot1 * dt
    state[:, 5] += rotation_speed * rot2 * dt
    # ADD ROTATION CLIPPING
    max_angle = np.deg2rad(drone_conf['th_max_angle'])
    state[:, 4] = np.clip(state[:, 4].real, -max_angle, max_angle)
    state[:, 5] = np.clip(state[:, 5].real, -max_angle, max_angle)

    # THRUST
    # magnitude
    thrust1: float = t1 * drone_conf['th_force']
    thrust2: float = t2 * drone_conf['th_force']

    # vector
    thrust1dir: np.ndarray = 1j * np.exp(1j * state[:, 4])
    thrust2dir: np.ndarray = 1j * np.exp(1j * state[:, 5])
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
    state[:, 3] += ang_acc * dt  # ang vel
    state[:, 2] += state[:, 3] * dt # ang rotation

    # translation
    F_world = F * np.exp(1j * state[:, 2])
    acc: np.ndarray = F_world / drone_conf['M'] - 9.81j

    vel: np.ndarray = acc * dt
    state[:, 1] += vel

    trans: np.ndarray = state[:, 1] * dt
    state[:, 0] += trans

    return state

def gen_target_chain(length, limit, dt, rng) -> np.ndarray:
    # purpose: to map every tick in sim to the position of target after touch
    rng = np.random.default_rng()
    n_segments = 5
    n_points   = n_segments + 1   # origin -> wp1 + motion segments

    # path length + speed -------------
    alpha = 3.0
    concentration = 15
    lengths = rng.dirichlet([alpha] * n_points)          # including origin -> wp1
    times   = rng.dirichlet(lengths[1:] * concentration) # NOT including origin -> wp1

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
    choice = rng.choice(len(maneuvers), size=n_segments, p=weights / weights.sum())

    # remove left/right bias
    sign = rng.choice([-1.0, 1.0], size=n_segments)
    deltas = sign * rng.vonmises(mu=mus[choice], kappa=kappas[choice])

    # add to angles
    angles = np.empty(n_points)
    angles[0] = rng.uniform(-np.pi, np.pi)              # initial heading is free
    angles[1:] = angles[0] + np.cumsum(deltas)

    # normalize --------------------------------------------------------------
    # motion fills n_segments / n_points of the timeline (origin->wp1 is non-moving)
    lengths = length * lengths
    times   = (n_segments / n_points) * limit * times

    # gen segment pos boundaries, n_segments -> n_points boundaries
    paths     = lengths * np.exp(1j * angles)
    waypoints = np.cumsum(paths)

    # gen segment time boundaries, n_segments -> n_points boundaries
    cumtime     = np.empty(n_points)
    cumtime[0]  = 0
    cumtime[1:] = np.cumsum(times)

    # generate timestamp for every tick in sim
    n_ticks = int(np.ceil(limit/dt))
    t       = np.arange(n_ticks) * dt

    # match every timestamps correspoinding segment, length = t range = 0..n_segments
    segment = np.searchsorted(cumtime, t, side='right') # gets index where sorted t would be inserted in cumtime
    segment = np.clip(segment - 1, 0, n_segments - 1)   # insertion idx - 1 = segment idx, clip to exclude end boundary

    # get completion percentage, (timestamp - segment start time) / segment time length
    # for every timestamp, length = t range = 0-1
    fraction = (t - cumtime[segment]) / times[segment]
    fraction = np.clip(fraction, 0, 1)

    # calculate lerp position in segment
    t_pos = (waypoints[segment+1] - waypoints[segment]) * fraction + waypoints[segment]

    return t_pos


# individuals + sim settings -> simulation stats
def sim(individuals: list[Individual], settings, seed=None) -> dict:
    # get configuration
    drone_conf = get_drone_conf(config_path)
    N = len(individuals)
    dt = .016

    # initialize targets
    rng = np.random.default_rng(seed)
    # get the target position every tick
    tick_pos: np.ndarray = gen_target_chain(settings['length'], settings['limit'], dt, rng)

    # init brain
    Brain = brain(individuals)

    # have physics states of all drones in one matrix
    # rows are drones, columns is state values
    # randomized init, identical across drones on this seed
    angle0   = rng.uniform(-np.deg2rad(15), np.deg2rad(15))
    ang_vel0 = rng.uniform(-0.5, 0.5)

    v_init_max = 1
    v_mag = np.sqrt(rng.uniform(0, 1)) * v_init_max   # sqrt -> uniform 2D disk
    v_dir = rng.uniform(-np.pi, np.pi)
    vel0  = v_mag * np.exp(1j * v_dir)

    state_matrix = np.zeros((N, 6), dtype=complex)
    state_matrix[:, 0] = 0j         # spawn at origin
    state_matrix[:, 1] = vel0
    state_matrix[:, 2] = angle0
    state_matrix[:, 3] = ang_vel0
    # columns 4, 5 (thruster angles) stay 0

    # initial action matrix
    action_matrix = np.zeros((N, 4), dtype=float)

    # simulation progression arrays
    toggle  = np.zeros(N, dtype=bool) # tracks initial touch to start chain
    ticks   = np.zeros(N, dtype=int) # ticks since touched

    # fitness + descriptor arrays
    # descriptors: mean | angular velocity |, mean thrust saturation
    fitness  = np.zeros(N)
    mean_av  = np.zeros(N)
    mean_sat = np.zeros(N)
    total_ticks = 0

    # prenitialize values
    prev_delta = None
    time = 0
    while time < settings['limit']:
        # UPDATE PHYSICS
        state_matrix = physics_update(dt, state_matrix, action_matrix, drone_conf)

        # MAKE OBSERVATION ARRAY -----------------------------------------------------------
        # obs values:
        #  1. delta pos x,  2. delta pos y
        #  3. old delta x,  4. old delta y
        #  5. velocity x,   6. volocity y
        #  7. sin(angle),   8. cos(angle)
        #  9. ang velocity
        # 10. t1 angle,    11. t2 angle

        # deltas
        target = tick_pos[ticks]
        delta_world   = target - state_matrix[:, 0]
        delta  = delta_world * np.exp(-1j * state_matrix[:, 2].real)
        if prev_delta is None:
            prev_delta = delta.copy()
        # velocity
        vel = state_matrix[:, 1] * np.exp(-1j * state_matrix[:, 2].real)
        # angles
        angle   = state_matrix[:, 2].real
        ang_vel = state_matrix[:, 3].real
        t1_ang  = state_matrix[:, 4].real
        t2_ang  = state_matrix[:, 5].real

        def slog(x):
            return np.sign(x) * np.log1p(np.abs(x))

        obs = np.column_stack([
            delta.real, delta.imag,
            prev_delta.real, prev_delta.imag,
            vel.real, vel.imag,
            np.sin(angle), np.cos(angle),
            ang_vel,
            t1_ang, t2_ang
        ])[:, :, np.newaxis] # have to have 3 dim for forward pass

        prev_delta = delta.copy()

        # FORWARD PASS OBSERVATIONS --------------------------------------------------------
        action_matrix = Brain.forward(obs)[:, :, 0] # getting rid of extra dim

        # PROGRESS SIMULATION --------------------------------------------------------------
        # check if touched waypoint
        dist = np.abs(delta_world)
        toggle |= dist < 0.5

        # give fitness for distance from target: 0.01x before touch, 1x after touch
        scale = np.where(toggle, 1.0, 0.01)
        score = scale * dt / ( 1 + dist )
        fitness += score

        # update descriptors
        mean_av  += np.abs(ang_vel)
        t1 = np.maximum(action_matrix[:, 0], 0)
        t2 = np.maximum(action_matrix[:, 1], 0)
        mean_sat += (np.maximum(t1, t2) > 0.9)

        # forward ticks it toggled
        ticks += toggle
        total_ticks += 1

        time += dt
    mean_av  /= total_ticks
    mean_sat /= total_ticks

    for i, ind in enumerate(individuals):
        ind.fitness = fitness[i]
        ind.descriptors = {'ang_vel': mean_av[i], 'saturation': mean_sat[i]}

    return {'fit_mean': fitness.mean(), 'fit_max': fitness.max()}

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
    print(f"sim time: {sim_time:.3f}s")
    print(f"per drone: {sim_time*1000/qty:.3f}ms")
    print(f"realtime factor: {limit/sim_time:.3f}x")
