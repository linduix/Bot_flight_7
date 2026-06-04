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
    # Returns (featured, sector_coords, present) where:
    #   featured[k]       — the k-th picked elite
    #   sector_coords[k]  — (row, col) in the 3x3 split for featured[k]
    #   present           — set of (row, col) sectors that had at least one elite
    R, C = grid.shape
    K = 3
    featured = []
    sector_coords = []
    present = set()
    for i in range(K):
        for j in range(K):
            block = grid[i*R//K:(i+1)*R//K, j*C//K:(j+1)*C//K]
            cands = [x for x in block.flat if x is not None]
            if cands:
                best = max(cands, key=lambda x: (x.fitness if x.fitness is not None else -np.inf))
                featured.append(best)
                sector_coords.append((i, j))
                present.add((i, j))
    return featured, sector_coords, present


def draw_sector_viz(screen, sector_coords, present, highlight, label_font):
    # 3x3 grid in the top-right corner. Highlighted sector filled yellow,
    # other present sectors outlined, missing sectors drawn dim.
    K = 3
    cell = 32
    pad = 2
    margin = 12
    size = K * cell + (K + 1) * pad
    x0 = SCREEN_W - size - margin
    y0 = margin

    # backdrop
    bg = pg.Surface((size, size), pg.SRCALPHA)
    bg.fill((0, 0, 0, 140))
    screen.blit(bg, (x0, y0))

    hl_sector = sector_coords[highlight] if 0 <= highlight < len(sector_coords) else None
    for i in range(K):
        for j in range(K):
            rx = x0 + pad + j * (cell + pad)
            ry = y0 + pad + i * (cell + pad)
            rect = pg.Rect(rx, ry, cell, cell)
            if (i, j) == hl_sector:
                pg.draw.rect(screen, (255, 220, 60), rect)
                pg.draw.rect(screen, (255, 255, 255), rect, 1)
            elif (i, j) in present:
                pg.draw.rect(screen, (60, 60, 70), rect)
                pg.draw.rect(screen, (160, 160, 180), rect, 1)
            else:
                pg.draw.rect(screen, (35, 35, 40), rect)
                pg.draw.rect(screen, (70, 70, 75), rect, 1)

    # caption underneath
    if hl_sector is not None:
        cap = label_font.render(
            f"#{highlight}  sector ({hl_sector[0]},{hl_sector[1]})",
            True, (220, 220, 230))
        screen.blit(cap, (x0, y0 + size + 4))


def showcase(individuals, sector_coords, present, dt=1/60):
    pg.init()
    screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
    pg.display.set_caption("Showcase — follow the mouse")
    clock  = pg.time.Clock()
    font   = pg.font.SysFont(None, 28)
    label_font = pg.font.SysFont(None, 20)

    drone_conf = get_drone_conf(config_path)
    max_a = 2 * drone_conf['th_force'] / drone_conf['M']   # control authority for zem_a/zev_a cap
    Brain = brain(individuals)
    N = len(individuals)
    S = 1  # one trial per drone in showcase mode

    drone_surf    = build_drone_surf(drone_conf['width'], drone_conf['height'], METERS_TO_PIXELS)
    thruster_surf = build_thruster_surf(drone_conf['height'] * 2, METERS_TO_PIXELS)

    # highlight is cycled manually by mouse click (left = next, right = prev).
    highlight = 0

    # init state matrix (N, 6, S)
    state_matrix  = np.zeros((N, 6, S), dtype=complex)
    action_matrix = np.zeros((N, 4, S), dtype=float)
    prev_los      = None
    prev_target   = None
    prev_vel      = None

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
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    highlight = (highlight + 1) % N
                elif event.button == 3:
                    highlight = (highlight - 1) % N

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

        # observations (27-input, mirrors sim1.py)
        eps   = 1e-8
        angle = state_matrix[:, 2, :].real

        delta_world = target - state_matrix[:, 0, :]
        delta_local = delta_world * np.exp(-1j * angle)

        # target geometry
        dist  = np.abs(delta_world)
        los_u = delta_world / (dist + eps)
        los_local = delta_local / (dist + eps)
        if prev_los is None:
            prev_los = los_u
        # los rate in LOCAL frame = inertial los rate - drone angular velocity
        los_rate = np.tanh((np.angle(los_u / prev_los) / dt - state_matrix[:, 3, :].real) / 3.0)
        prev_los = los_u

        # self velocity (local)
        vel = state_matrix[:, 1, :] * np.exp(-1j * angle)
        vel_mag = np.abs(vel); vel_u = vel / (vel_mag + eps)

        # relative velocity (local), target vel from frame-to-frame mouse motion
        if prev_target is None:
            prev_target = target
        v_target = (target - prev_target) / dt
        prev_target = target
        v_target_local = v_target * np.exp(-1j * angle)
        rel_vel = vel - v_target_local
        relvel_mag = np.abs(rel_vel); relvel_u = rel_vel / (relvel_mag + eps)

        # prev-tick net acceleration (world -> local)
        vel_world = state_matrix[:, 1, :].copy()   # copy: state_matrix is mutated in place next tick
        if prev_vel is None:
            prev_vel = vel_world
        net_acc = ((vel_world - prev_vel) / dt) * np.exp(-1j * angle)
        prev_vel = vel_world
        acc_mag = np.abs(net_acc); acc_u = net_acc / (acc_mag + eps)

        # time to closest approach
        closest_approach = (delta_local * np.conjugate(rel_vel)).real / (np.abs(rel_vel) + eps)
        tti_raw = closest_approach / (np.abs(rel_vel) + eps)
        tti_obs = np.tanh(tti_raw / 10.0)

        # guidance accel commands: ZEM/ZEV at t_go = tti_raw w/ gravity, PN form, mag-capped
        grav_body = (-1j * drone_conf['G']) * np.exp(-1j * angle)
        zem = delta_local - rel_vel * tti_raw + 0.5 * grav_body * tti_raw ** 2
        zev = v_target_local - vel + grav_body * tti_raw
        zem_a = zem / (tti_raw ** 2 + eps)
        zev_a = zev / (tti_raw + eps)
        zem_a = zem_a / (np.abs(zem_a) + eps) * np.tanh(np.abs(zem_a) / max_a)
        zev_a = zev_a / (np.abs(zev_a) + eps) * np.tanh(np.abs(zev_a) / max_a)
        zema_mag = np.abs(zem_a); zema_u = zem_a / (zema_mag + eps)
        zeva_mag = np.abs(zev_a); zeva_u = zev_a / (zeva_mag + eps)

        # attitude / actuators
        ang_vel   = state_matrix[:, 3, :].real
        t1_ang    = state_matrix[:, 4, :].real
        t2_ang    = state_matrix[:, 5, :].real
        t1_thrust = action_matrix[:, 0, :]
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
        ], axis=1)

        action_matrix = Brain.forward(obs)

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
            draw_drone(screen, state_t0[i], drone_surf, thruster_surf, drone_conf, alpha=55)
            px, py = world_to_screen(state_t0[i, 0])
            lab = label_font.render(f"#{i}", True, (110, 110, 140))
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
        screen.blit(label_font.render("click to cycle drones (right-click = prev) — ESC to quit",
                                       True, (130, 130, 130)), (10, 36))

        # sector viz (top-right)
        draw_sector_viz(screen, sector_coords, present, highlight, label_font)

        pg.display.flip()
        clock.tick(60)

    pg.quit()


if __name__ == "__main__":
    save_path = os.path.join('data', 'MAP_Checkpoint.pkl')
    alg, settings, _ = load(save_path)

    featured, sector_coords, present = pick_featured(alg.archive.indv)
    if not featured:
        print("No elites found in archive.")
        raise SystemExit(1)

    print(f"loaded {len(featured)} featured elites from {save_path} (gen {alg.gen})")
    showcase(featured, sector_coords, present)
