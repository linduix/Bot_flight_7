import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from modules.individual import Individual
from modules.batch_brain import brain
from modules.simulation.sim1 import physics_update, get_drone_conf, config_path, gen_target_chain
import numpy as np
import pygame as pg


METERS_TO_PIXELS = 30
SCREEN_W = 1280
SCREEN_H = 720


def world_to_screen(pos_complex: complex) -> tuple[int, int]:
    x = pos_complex.real * METERS_TO_PIXELS + SCREEN_W / 2
    y = SCREEN_H / 2 - pos_complex.imag * METERS_TO_PIXELS
    return int(x), int(y)


def build_drone_surf(width_m, height_m, mtp):
    w = int(width_m * mtp)
    h = int(height_m * mtp)
    scale = 10
    pad = int(h * 1.75 * 2)
    big = pg.Surface((w * scale, pad * scale), pg.SRCALPHA)

    pg.draw.rect(big, (220, 220, 220),
                 (0, int((pad // 2 - h // 2) * scale), w * scale, h * scale))
    pg.draw.circle(big, (220, 220, 220),
                   (w * scale // 2, pad * scale // 2), int(h * 1.75 * scale))
    pg.draw.circle(big, (230, 100, 100),
                   (w * scale // 2, int((pad // 2 - h / 1.5) * scale)), int(h * 0.4 * scale))

    return pg.transform.smoothscale(big, (w, pad))


def build_thruster_surf(size_m, mtp):
    s = int(size_m * mtp)
    scale = 10
    big = pg.Surface((s * scale, s * scale), pg.SRCALPHA)
    pts = [(0, 0), (s * scale, 0),
           (s * scale * 3 // 4, s * scale),
           (s * scale * 1 // 4, s * scale)]
    pg.draw.polygon(big, (175, 175, 175), pts)
    return pg.transform.smoothscale(big, (s, s))


def draw_drone(screen, state_row, drone_surf, thruster_surf, drone_conf, alpha=255):
    pos   = state_row[0]
    angle = state_row[2].real
    t1ang = state_row[4].real
    t2ang = state_row[5].real

    cx, cy = world_to_screen(pos)

    # body
    rotated = pg.transform.rotate(drone_surf, np.rad2deg(angle))
    rotated.set_alpha(alpha)
    screen.blit(rotated, rotated.get_rect(center=(cx, cy)))

    # thruster offset in world frame
    offset = drone_conf['th_offset'] * np.exp(1j * angle)

    t1pos = pos - offset
    t2pos = pos + offset

    t1x, t1y = world_to_screen(t1pos)
    t2x, t2y = world_to_screen(t2pos)

    t1_rot = pg.transform.rotate(thruster_surf, np.rad2deg(t1ang + angle))
    t1_rot.set_alpha(alpha)
    screen.blit(t1_rot, t1_rot.get_rect(center=(t1x, t1y)))

    t2_rot = pg.transform.rotate(thruster_surf, np.rad2deg(t2ang + angle))
    t2_rot.set_alpha(alpha)
    screen.blit(t2_rot, t2_rot.get_rect(center=(t2x, t2y)))


class ParticlePool:
    def __init__(self, max_particles=2000):
        self.max = max_particles
        self.pos  = np.zeros(max_particles, dtype=complex)
        self.vel  = np.zeros(max_particles, dtype=complex)
        self.life = np.zeros(max_particles)
        self.max_life = np.zeros(max_particles)
        self.start_alpha = np.zeros(max_particles)
        self._next = 0

    def spawn(self, pos: complex, vel: complex, lifetime=0.25, start_alpha=255):
        i = self._next % self.max
        self.pos[i]         = pos
        self.vel[i]         = vel
        self.life[i]        = lifetime
        self.max_life[i]    = lifetime
        self.start_alpha[i] = start_alpha
        self._next += 1

    def update(self, dt):
        alive = self.life > 0
        self.pos[alive]  += self.vel[alive] * dt
        self.life[alive] -= dt

    def draw(self, screen, radius=3):
        alive = np.where(self.life > 0)[0]
        for i in alive:
            alpha = int(self.start_alpha[i] * self.life[i] / self.max_life[i])
            sx, sy = world_to_screen(self.pos[i])
            surf = pg.Surface((radius * 2, radius * 2), pg.SRCALPHA)
            pg.draw.circle(surf, (255, 150, 50, alpha), (radius, radius), radius)
            screen.blit(surf, (sx - radius, sy - radius))


def spawn_thruster_particles(pool: ParticlePool, state_matrix, action_matrix, drone_conf, n):
    for i in range(n):
        t1_thrust = action_matrix[i, 0]
        t2_thrust = action_matrix[i, 1]
        pos   = state_matrix[i, 0]
        angle = state_matrix[i, 2].real
        t1ang = state_matrix[i, 4].real
        t2ang = state_matrix[i, 5].real
        drone_vel = state_matrix[i, 1]
        ang_vel   = state_matrix[i, 3].real

        offset = drone_conf['th_offset'] * np.exp(1j * angle)

        # thruster velocities = drone center velocity + rotational contribution
        t1_vel = drone_vel + 1j * ang_vel * (-offset)
        t2_vel = drone_vel + 1j * ang_vel * (+offset)

        spawn_offset = 0.1

        if t1_thrust > 0.05:
            world_ang = t1ang + angle
            direction = -1j * np.exp(1j * world_ang)
            t1pos = pos - offset + direction * spawn_offset
            for _ in range(2):
                speed = np.random.uniform(7, 12) * t1_thrust
                vel   = direction * speed + t1_vel
                pool.spawn(t1pos, vel, start_alpha=255 * t1_thrust)

        if t2_thrust > 0.05:
            world_ang = t2ang + angle
            direction = -1j * np.exp(1j * world_ang)
            t2pos = pos + offset + direction * spawn_offset
            for _ in range(2):
                speed = np.random.uniform(7, 12) * t2_thrust
                vel   = direction * speed + t2_vel
                pool.spawn(t2pos, vel, start_alpha=255 * t2_thrust)


def sim(individuals: list[Individual], settings, seed=None) -> dict:
    # highlight highest-fitness drone if all have a saved fitness
    fits = [ind.fitness for ind in individuals]
    highlight = int(np.argmax(fits)) if all(f is not None for f in fits) else None
    pg.init()
    screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
    pg.display.set_caption("Sim1 Visual")
    clock = pg.time.Clock()
    font = pg.font.SysFont(None, 28)

    drone_conf = get_drone_conf(config_path)
    Brain = brain(individuals)
    N = len(individuals)

    drone_surf    = build_drone_surf(drone_conf['width'], drone_conf['height'], METERS_TO_PIXELS)
    thruster_surf = build_thruster_surf(drone_conf['height'] * 2, METERS_TO_PIXELS)

    speed_factor = 1
    frame_dt = 1 / 60
    dt = frame_dt / speed_factor

    # generated target chain (same logic as headless sim)
    rng = np.random.default_rng(seed)
    tick_pos: np.ndarray = gen_target_chain(settings['length'], settings['limit'], dt, rng)

    # randomized init, identical across drones (match sim1 for fair swap-in)
    angle0   = rng.uniform(-np.deg2rad(15), np.deg2rad(15))
    ang_vel0 = rng.uniform(-0.5, 0.5)
    v_init_max = 1
    v_mag    = np.sqrt(rng.uniform(0, 1)) * v_init_max
    v_dir    = rng.uniform(-np.pi, np.pi)
    vel0     = v_mag * np.exp(1j * v_dir)

    state_matrix = np.zeros((N, 6), dtype=complex)
    state_matrix[:, 0] = 0j
    state_matrix[:, 1] = vel0
    state_matrix[:, 2] = angle0
    state_matrix[:, 3] = ang_vel0

    action_matrix = np.zeros((N, 4), dtype=float)

    # simulation state arrays
    toggle = np.zeros(N, dtype=bool)
    ticks  = np.zeros(N, dtype=int)
    particles = ParticlePool(max_particles=4000)
    prev_delta = None

    # fitness + descriptor accumulators
    fitness  = np.zeros(N)
    mean_av  = np.zeros(N)
    mean_sat = np.zeros(N)
    total_ticks = 0

    sim_time = 0.0
    quit_early = False
    while sim_time < settings['limit']:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                quit_early = True
                break
        if quit_early:
            break

        # UPDATE PHYSICS
        state_matrix = physics_update(dt, state_matrix, action_matrix, drone_conf)

        # per-drone target from generated chain (clamp to last tick)
        idx = np.minimum(ticks, len(tick_pos) - 1)
        target = tick_pos[idx]

        # MAKE OBSERVATION ARRAY
        delta_world = target - state_matrix[:, 0]
        delta = delta_world * np.exp(-1j * state_matrix[:, 2].real)
        if prev_delta is None:
            prev_delta = delta.copy()
        vel     = state_matrix[:, 1] * np.exp(-1j * state_matrix[:, 2].real)
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
        ])[:, :, np.newaxis]

        prev_delta = delta.copy()

        action_matrix = Brain.forward(obs)[:, :, 0]

        # PROGRESS SIMULATION
        dist = np.abs(delta_world)
        toggle |= dist < 0.5

        # fitness: 0.01x pre-touch, 1x post-touch
        scale = np.where(toggle, 1.0, 0.01)
        fitness += scale * dt / (1 + dist)

        # descriptors
        mean_av += np.abs(ang_vel)
        t1 = np.maximum(action_matrix[:, 0], 0)
        t2 = np.maximum(action_matrix[:, 1], 0)
        mean_sat += (np.maximum(t1, t2) > 0.9)

        ticks += toggle
        total_ticks += 1
        sim_time += dt

        # particles + draw
        spawn_thruster_particles(particles, state_matrix, action_matrix, drone_conf, N)
        particles.update(dt)

        screen.fill((20, 20, 20))
        particles.draw(screen)

        pg.draw.line(screen, (200, 60, 60),
                     world_to_screen(0j), world_to_screen(tick_pos[0]), 1)

        path_pts = [world_to_screen(p) for p in tick_pos[::max(1, len(tick_pos)//200)]]
        if len(path_pts) > 1:
            pg.draw.lines(screen, (60, 100, 60), False, path_pts, 1)
        for i in range(N):
            tx, ty = world_to_screen(target[i])
            pg.draw.circle(screen, (100, 230, 100), (tx, ty), 3)

        for i in range(N):
            if i == highlight:
                continue   # draw highlighted last so it sits on top
            draw_drone(screen, state_matrix[i], drone_surf, thruster_surf, drone_conf, alpha=100)
        if highlight is not None:
            hx, hy = world_to_screen(state_matrix[highlight, 0])
            pg.draw.circle(screen, (255, 220, 60), (hx, hy), 18, 2)
            draw_drone(screen, state_matrix[highlight], drone_surf, thruster_surf, drone_conf, alpha=255)

        fps = clock.get_fps()
        screen.blit(font.render(f"FPS: {fps:.0f}  Drones: {N}  t={sim_time:.1f}/{settings['limit']:.1f}", True, (150, 150, 150)), (10, 10))

        pg.display.flip()
        clock.tick(60)

    pg.quit()

    if total_ticks > 0:
        mean_av  /= total_ticks
        mean_sat /= total_ticks

    for i, ind in enumerate(individuals):
        ind.fitness = fitness[i]
        ind.descriptors = {'ang_vel': mean_av[i], 'saturation': mean_sat[i]}

    return {'fit_mean': fitness.mean(), 'fit_max': fitness.max()}


if __name__ == "__main__":
    import os
    from modules.evo_alg.mapElites import load

    save_path = os.path.join('data', 'MAP_Checkpoint.pkl')
    alg, settings, _ = load(save_path)
    # grid-aligned stride-4 sample of the archive (4^2 = 16x reduction)
    elites = [x for x in alg.archive.indv[::4, ::4].flat if x is not None]
    print(f"loaded {len(elites)} elites from {save_path} (gen {alg.gen})")
    print(f"settings: {settings}")
    sim(elites, settings)
