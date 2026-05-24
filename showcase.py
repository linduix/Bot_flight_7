import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import numpy as np
import pygame as pg

from modules.batch_brain import brain
from modules.simulation.sim1 import physics_update, get_drone_conf, config_path
from modules.simulation.sim1_visual import (
    ParticlePool, build_drone_surf, build_thruster_surf,
    draw_drone, spawn_thruster_particles, world_to_screen,
    METERS_TO_PIXELS, SCREEN_W, SCREEN_H,
)
from modules.evo_alg.mapElites import load


def screen_to_world(mx: int, my: int) -> complex:
    x = (mx - SCREEN_W / 2) / METERS_TO_PIXELS
    y = (SCREEN_H / 2 - my) / METERS_TO_PIXELS
    return complex(x, y)


def pick_featured(grid):
    # Pick the best elite from each cell of a 3x3 split of the archive grid.
    R, C = grid.shape
    K = 3
    featured = []
    seen = set()
    for i in range(K):
        for j in range(K):
            block = grid[i*R//K:(i+1)*R//K, j*C//K:(j+1)*C//K]
            cands = [x for x in block.flat if x is not None]
            if cands:
                best = max(cands, key=lambda x: (x.fitness if x.fitness is not None else -np.inf))
                featured.append(best)
                seen.add(id(best))
    return featured


def showcase(individuals, dt=1/60):
    pg.init()
    screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
    pg.display.set_caption("Showcase — follow the mouse")
    clock  = pg.time.Clock()
    font   = pg.font.SysFont(None, 28)
    label_font = pg.font.SysFont(None, 20)

    drone_conf = get_drone_conf(config_path)
    Brain = brain(individuals)
    N = len(individuals)
    S = 1  # one trial per drone in showcase mode

    drone_surf    = build_drone_surf(drone_conf['width'], drone_conf['height'], METERS_TO_PIXELS)
    thruster_surf = build_thruster_surf(drone_conf['height'] * 2, METERS_TO_PIXELS)

    # live score = slow EMA of 1/(1+dist); highlight switches only when a challenger
    # beats the current leader by `switch_margin` (hysteresis prevents flicker).
    live_score    = np.zeros(N)
    ema_alpha     = 0.01    # ~100-frame time constant
    switch_margin = 1.10    # challenger must be 10% better to take over
    highlight     = 0

    # init state matrix (N, 6, S)
    state_matrix  = np.zeros((N, 6, S), dtype=complex)
    action_matrix = np.zeros((N, 4, S), dtype=float)
    prev_delta    = None

    # distance after which we respawn a drone (offscreen)
    reset_dist = np.hypot(SCREEN_W, SCREEN_H) / METERS_TO_PIXELS

    particles = ParticlePool(max_particles=4000)

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                running = False

        # target = mouse position
        mx, my = pg.mouse.get_pos()
        target_c = screen_to_world(mx, my)
        target = np.full((N, S), target_c, dtype=complex)

        # physics
        state_matrix = physics_update(dt, state_matrix, action_matrix, drone_conf)

        # respawn drones that drift too far
        pos_t0 = state_matrix[:, 0, 0]
        far = np.abs(pos_t0 - target_c) > reset_dist
        if np.any(far):
            for i in np.where(far)[0]:
                state_matrix[i, :, :] = 0
                state_matrix[i, 2, :] = np.random.uniform(-np.deg2rad(15), np.deg2rad(15))
            # crashing (drifted offscreen → respawn) halves the live score
            live_score[far] *= 0.5

        # observations
        delta_world = target - state_matrix[:, 0, :]
        delta = delta_world * np.exp(-1j * state_matrix[:, 2, :].real)
        if prev_delta is None:
            prev_delta = delta.copy()
        vel     = state_matrix[:, 1, :] * np.exp(-1j * state_matrix[:, 2, :].real)
        angle   = state_matrix[:, 2, :].real
        ang_vel = state_matrix[:, 3, :].real
        t1_ang  = state_matrix[:, 4, :].real
        t2_ang  = state_matrix[:, 5, :].real

        obs = np.stack([
            delta.real, delta.imag,
            prev_delta.real, prev_delta.imag,
            vel.real, vel.imag,
            np.sin(angle), np.cos(angle),
            ang_vel,
            t1_ang, t2_ang,
        ], axis=1)
        prev_delta = delta.copy()

        action_matrix = Brain.forward(obs)

        # update live score + highlight
        inst = 1.0 / (1.0 + np.abs(delta_world[:, 0]))
        live_score = (1 - ema_alpha) * live_score + ema_alpha * inst
        challenger = int(np.argmax(live_score))
        if live_score[challenger] > live_score[highlight] * switch_margin:
            highlight = challenger

        # ---- RENDER ----
        state_t0  = state_matrix[:, :, 0]
        action_t0 = action_matrix[:, :, 0]

        spawn_thruster_particles(particles, state_t0, action_t0, drone_conf, [highlight])
        particles.update(dt)

        screen.fill((20, 20, 20))
        particles.draw(screen)

        # draw non-highlighted featured drones with low alpha
        for i in range(N):
            if i == highlight:
                continue
            draw_drone(screen, state_t0[i], drone_surf, thruster_surf, drone_conf, alpha=140)
            px, py = world_to_screen(state_t0[i, 0])
            lab = label_font.render(f"#{i}", True, (180, 180, 220))
            screen.blit(lab, lab.get_rect(center=(px, py - 30)))

        # highlight drone (no ring)
        hx, hy = world_to_screen(state_t0[highlight, 0])
        draw_drone(screen, state_t0[highlight], drone_surf, thruster_surf, drone_conf, alpha=255)
        lab = label_font.render(f"BEST #{highlight}", True, (255, 220, 60))
        screen.blit(lab, lab.get_rect(center=(hx, hy - 30)))

        # target marker (mouse)
        pg.draw.circle(screen, (100, 230, 100), (mx, my), 4)

        # HUD
        screen.blit(font.render(f"FPS: {clock.get_fps():.0f}  Drones: {N}", True, (150, 150, 150)),
                    (10, 10))
        screen.blit(label_font.render("move mouse to set target — ESC to quit",
                                       True, (130, 130, 130)), (10, 36))

        pg.display.flip()
        clock.tick(60)

    pg.quit()


if __name__ == "__main__":
    save_path = os.path.join('data', 'MAP_Checkpoint.pkl')
    alg, settings, _ = load(save_path)

    featured = pick_featured(alg.archive.indv)
    if not featured:
        print("No elites found in archive.")
        raise SystemExit(1)

    print(f"loaded {len(featured)} featured elites from {save_path} (gen {alg.gen})")
    showcase(featured)
