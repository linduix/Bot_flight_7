from modules.individual import Individual
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
        M : drone_config['mass'],
        I : drone_config['inertia'],
        width : drone_config['width'],
        height : drone_config['height'],
        G : drone_config['gravity'],
        th_offset : drone_config['thruster_offset'],
        th_rotation_speed : drone_config['thruster_rotation_speed'],
        th_max_angle : drone_config['thruster_max_angle'],
        th_force : drone_config['thruster_force']
    }
    return drone


# physics state x + i*y, vx + i*vx, angle, angular vel, t1 angle, t2 angle
state = [0+0j, 0+0j, 0, 0, 0]

def sim(individuals: list[Individual], settings) -> dict:
    brains = []
    for i in individuals:
        b = brain(i.weights)
        brains.append(b)


    return {}

if __name__=="__main__":
    config_path = '..\\..\\config.toml'
    config_path = 'config.toml'
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
        drone_config = config['drone']

    M = drone_config['mass']
    I = drone_config['inertia']
    width = drone_config['width']
    height = drone_config['height']
    G = drone_config['gravity']
    th_offset = drone_config['thruster_offset']
    th_rotation_speed = drone_config['thruster_rotation_speed']
    th_max_angle = drone_config['thruster_max_angle']
    th_force = drone_config['thruster_force']
