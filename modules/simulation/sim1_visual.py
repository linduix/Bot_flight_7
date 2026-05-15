import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from modules.individual import Individual
from modules.batch_brain import brain
from modules.simulation.sim1 import physics_update, get_drone_conf, config_path
import numpy as np
import pygame as pg


METERS_TO_PIXELS = 20
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


def run_visual(individuals: list[Individual]):
    pg.init()
    screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
    pg.display.set_caption("Sim1 Visual")
    clock = pg.time.Clock()
    font = pg.font.SysFont(None, 28)

    drone_conf = get_drone_conf(config_path)
    Brain = brain(individuals)
    n = len(individuals)

    drone_surf    = build_drone_surf(drone_conf['width'], drone_conf['height'], METERS_TO_PIXELS)
    thruster_surf = build_thruster_surf(drone_conf['height'] * 2, METERS_TO_PIXELS)

    state_matrix  = np.zeros((n, 6), dtype=complex)
    action_matrix = np.zeros((n, 4), dtype=float)

    target = np.zeros(n, dtype=complex)
    particles = ParticlePool(max_particles=4000)

    dt = 1 / 60

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return

        # mouse position → world space
        mx, my = pg.mouse.get_pos()
        wx = (mx - SCREEN_W / 2) / METERS_TO_PIXELS
        wy = (SCREEN_H / 2 - my) / METERS_TO_PIXELS
        target[:] = wx + 1j * wy

        # observations (zeros for now — wire up properly when obs are ready)
        obs = np.zeros((n, 11, 1))
        action_matrix = Brain.forward(obs)[:, :, 0]

        speed_factor = 5
        state_matrix = physics_update(dt/speed_factor, state_matrix, action_matrix, drone_conf)
        spawn_thruster_particles(particles, state_matrix, action_matrix, drone_conf, n)
        particles.update(dt/speed_factor)

        screen.fill((20, 20, 20))

        particles.draw(screen)

        # draw target
        tx, ty = world_to_screen(target[0])
        pg.draw.circle(screen, (100, 230, 100), (tx, ty), 5)

        for i in range(n):
            draw_drone(screen, state_matrix[i], drone_surf, thruster_surf, drone_conf, alpha=180)

        fps = clock.get_fps()
        screen.blit(font.render(f"FPS: {fps:.0f}  Drones: {n}", True, (150, 150, 150)), (10, 10))

        pg.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    from modules.evo_alg.stub import evostub
    alg = evostub()
    indv, _ = alg.propose(100, None)
    run_visual(indv)
