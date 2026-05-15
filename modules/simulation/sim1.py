from modules.individual import Individual
import numpy as np
import tomllib

class brain():
    def __init__(self, weights) -> None:
        pass
    def forward(self, obs):
        return [0, 0, 0, 0]

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

# physics state + actions -> updated state
def physics_update(dt, state: list, actions: list, drone_conf: dict):
    # process actions
    t1, t2, rot1, rot2 = actions[0], actions[1], actions[2], actions[3]
    t1, t2 = max(0, t1), max(0, t2)

    # calculate forces (drone refrence frame) --------------------
    rotation_speed = drone_conf['th_rotation_speed']
    # thruster rotation
    state[4] += rotation_speed * rot1 * dt
    state[5] += rotation_speed * rot2 * dt
    # ADD ROTATION CLIPPING

    # THRUST
    # magnitude
    thrust1: float = t1 * drone_conf['th_force']
    thrust2: float = t2 * drone_conf['th_force']

    # vector
    thrust1dir: complex = 1j * np.exp(1j * state[4])
    thrust2dir: complex = 1j * np.exp(1j * state[5])
    F1: complex = thrust1 * thrust1dir
    F2: complex = thrust2 * thrust2dir
    F = F1 + F2

    # TORQUE
    # magnitude
    tau1: float = -drone_conf['thruster_offset'] * F1.imag
    tau2: float = drone_conf['thruster_offset'] * F2.imag
    T = tau1 + tau2

    # update physics (world frame) -------------------------------
    # rotate drone
    ang_acc   = T / drone_conf['I']
    state[3] += ang_acc * dt  # ang vel
    state[2] += state[3] * dt # ang rotation

    # translation
    F_world = F * np.exp(1j * state[2])
    acc: complex = F_world / drone_conf['M'] - 9.81j

    vel: complex = acc * dt
    state[1] += vel

    trans: complex = state[1] * dt
    state[0] += trans

    return state

# individuals + sim settings -> simulation stats
def sim(individuals: list[Individual], settings) -> dict:
    brains = []
    for i in individuals:
        b = brain(i.weights)
        brains.append(b)

    time = 0
    dt = .016
    while time < 10:
        for b, indv in zip(brains, individuals):
            # make observations array
            # forward pass obsrv to brain
            # update physics state using output
            # add to fitness
            pass

        time += dt


    return {}

if __name__=="__main__":
    config_path = '..\\..\\config.toml'
    drone_conf = get_drone_conf(config_path)
