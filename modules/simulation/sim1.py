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

# individuals + sim settings -> simulation stats
def sim(individuals: list[Individual], settings) -> dict:
    # get configuration
    drone_conf = get_drone_conf(config_path)
    N = len(individuals)

    # initialize targets
    target = 0 + 0j
    
    # set brain
    Brain = brain(individuals)

    # have states of all drones in one matrix
    # rows are drones, columns is state values
    state_matrix = np.zeros((N, 6), dtype=complex)
    action_matrix = np.zeros((N, 4), dtype=float)

    # prenitialize values
    prev_delta = None

    time = 0
    dt = .016
    while time < settings['limit']:
        # update physics for all drones
        state_matrix = physics_update(dt, state_matrix, action_matrix, drone_conf)

        # make observations array
        # obs values:
        #  1. delta pos x,  2. delta pos y
        #  3. old delta x,  4. old delta y
        #  5. velocity x,   6. volocity y
        #  7. sin(angle),   8. cos(angle)
        #  9. ang velocity
        # 10. t1 angle,    11. t2 angle 

        # deltas
        delta = (target - state_matrix[:, 0]) * np.exp(-1j * state_matrix[:, 2].real)
        if prev_delta is None:
            prev_delta = delta.copy()
        # velocity
        vel = state_matrix[:, 1] * np.exp(-1j * state_matrix[:, 2].real)
        # angles
        angle   = state_matrix[:, 2].real
        ang_vel = state_matrix[:, 3].real
        t1_ang  = state_matrix[:, 4].real
        t2_ang  = state_matrix[:, 5].real

        obs = np.column_stack([
            delta.real, delta.imag,
            prev_delta.real, prev_delta.imag,
            vel.real, vel.imag,
            np.sin(angle), np.cos(angle),
            ang_vel,
            t1_ang, t2_ang
        ])[:, :, np.newaxis] # have to have 3 dim for forward pass

        prev_delta = delta.copy()

        # forward pass obsrv to brain
        action_matrix = Brain.forward(obs)[:, :, 0] # getting rid of extra dim
        
        for indv in individuals:
            # add to fitness
            pass
        
        print(np.abs(state_matrix[:5, 1]))
        time += dt


    return {}

if __name__=="__main__":
    import time
    from modules.evo_alg.stub import evostub

    alg = evostub()
    qty = 300
    indv, _ = alg.propose(qty, None)

    limit = 1
    t0 = time.perf_counter()
    sim(indv, {'limit': limit})
    sim_time = time.perf_counter() - t0
    print(f"sim time: {sim_time:.3f}s")
    print(f"per drone: {sim_time*1000/qty:.3f}ms")
    print(f"realtime factor: {limit/sim_time:.3f}x")

