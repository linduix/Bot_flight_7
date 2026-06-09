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

# playback slowdown factor: 1 = normal speed, 2 = half speed, 4 = quarter speed, etc.
# only affects how fast it plays on screen — physics still steps at the training dt.
PLAYBACK_SLOWDOWN = 1.0


# world point the camera is centered on (complex, world meters). Updated each frame
# to follow the highlighted drone so it stays in the middle of the screen.
CAMERA = 0j

# seed mode: False = replay the checkpoint seed every loop, True = fresh random seed every loop.
# toggled at runtime by pressing T inside the visual. takes effect on the NEXT playback.
RANDOM_SEED_MODE = False

# camera follow mode: True = camera tracks the highlighted drone, False = camera tracks
# its current target. toggled with F inside the visual. takes effect on the SAME tick.
FOLLOW_DRONE = True

# scoring arrow overlay: True = draw the per-tick scoring vectors (ideal_vel, current vel,
# ideal_dv, actual_dv) + their text labels. toggled with A inside the visual.
SHOW_ARROWS = True

# ghost overlay: True = draw non-highlighted drones (dots + faint sprites) and the
# yellow highlight ring. False = show only the highlighted drone (no ring, no ghosts).
# toggled with G inside the visual.
SHOW_GHOSTS = True


def world_to_screen(pos_complex: complex) -> tuple[int, int]:
    rel = pos_complex - CAMERA
    x = rel.real * METERS_TO_PIXELS + SCREEN_W / 2
    y = SCREEN_H / 2 - rel.imag * METERS_TO_PIXELS
    return int(x), int(y)


def draw_vector(screen, base: complex, vec: complex, color, scale=0.4, width=2, head=0.35):
    # draw `vec` (a velocity in m/s, complex) as an arrow rooted at world point `base`.
    # `scale` is a seconds-like factor so the arrow length = |vec| * scale meters.
    if abs(vec) < 1e-6:
        return
    tip = base + vec * scale
    bx, by = world_to_screen(base)
    tx, ty = world_to_screen(tip)
    pg.draw.line(screen, color, (bx, by), (tx, ty), width)
    back = -vec / abs(vec)                     # unit dir from tip back toward base
    for a in (0.5, -0.5):                       # barbs splayed +/-0.5rad off the backward dir
        barb = tip + head * back * np.exp(1j * a)
        pg.draw.line(screen, color, (tx, ty), world_to_screen(barb), width)


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
            sx, sy = world_to_screen(self.pos[i]) # type:ignore
            surf = pg.Surface((radius * 2, radius * 2), pg.SRCALPHA)
            pg.draw.circle(surf, (255, 150, 50, alpha), (radius, radius), radius)
            screen.blit(surf, (sx - radius, sy - radius))


def spawn_thruster_particles(pool: ParticlePool, state_matrix, action_matrix, drone_conf, indices):
    for i in indices:
        # actual integrated throttle level (slot 6), not the rate command
        t1_thrust = state_matrix[i, 6].real
        t2_thrust = state_matrix[i, 6].imag
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
                pool.spawn(t2pos, vel, start_alpha=100 * t2_thrust)


def sim(individuals: list[Individual], settings, seed=None, featured=None, update_fitness=True) -> dict:
    # `featured` = indices drawn as full sprites; the rest render as ghost lines.
    # `highlight` (single) = max-fitness among featured; gets ring + thruster particles.
    fits = [ind.fitness for ind in individuals]
    if featured is None:
        featured = set(range(len(individuals)))
    else:
        featured = set(featured)
    if featured and all(fits[i] is not None for i in featured):
        highlight = max(featured, key=lambda i: fits[i]) # type:ignore
    elif featured:
        highlight = int(np.random.choice(list(featured)))
    else:
        highlight = int(np.random.randint(len(individuals)))
    pg.init()
    screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
    pg.display.set_caption("Sim1 Visual")
    clock = pg.time.Clock()
    font = pg.font.SysFont(None, 28)

    drone_conf = get_drone_conf(config_path)
    Brain = brain(individuals)
    N = len(individuals)
    S = 4  # trials per drone — only trial 0 is rendered; rest run "headlessly" so chains can be shown faintly

    # physics MUST step at the same dt the controller was trained on (sim1.py dt=.016),
    # otherwise the high-gain policy mistimes and flies wonky. render still caps at 60fps;
    # playback ends up ~0.96x real-time, which is fine.
    dt = 0.016

    # fitness constants (mirror sim1.py)
    max_a = 2 * drone_conf['th_force'] / drone_conf['M']
    eps   = 1e-8
    eps_d = 0.05
    floor = 0.5    # hover tolerance for the tracking scale, m/s
    effort_floor = 0.15  # min divisor for effort score so near-zero err_v doesn't demand impossible precision

    # scale shape: f(x) = 1/(K·(x+A)) − A, mirrors sim1.py.
    SCALE_A = 1.0 / 15.0
    SCALE_K = 225.0 / 16.0
    inv_L   = 1.0 / settings['length']

    # criticality gate for effort (mirrors sim1.py). quadratic exponent,
    # ~closed at hover, ~open at err_mag = 6·max_a·dt.
    crit_ref  = 6.0 * max_a * dt
    crit_coef = 6.0 * np.log(2.0) / (crit_ref ** 2)

    drone_surf    = build_drone_surf(drone_conf['width'], drone_conf['height'], METERS_TO_PIXELS)
    thruster_surf = build_thruster_surf(drone_conf['height'] * 2, METERS_TO_PIXELS)

    # generated target chains (one per trial, same logic as headless sim)
    rng = np.random.default_rng(seed)
    tick_pos: np.ndarray = gen_target_chain(settings['length'], settings['limit'], dt, rng, S)  # (S, n_ticks)

    # perturbations flag (settings) gates init randomization + wind gusts. when False,
    # drones spawn at origin with zero velocity/angle/spin and no gusts.
    perturbations = settings.get('perturbations', False)

    state_matrix = np.zeros((N, 7, S), dtype=complex)
    state_matrix[:, 0, :] = 0j
    # pre-spool throttles to hover (mirror sim1.py) so rate-command policies start viable
    hover = drone_conf['M'] * abs(drone_conf['G']) / (2 * drone_conf['th_force'])
    state_matrix[:, 6, :] = hover + 1j * hover

    if perturbations:
        # randomized init, identical across drones AND trials here (visual only — only trial 0 displays)
        angle0   = rng.uniform(-np.deg2rad(60), np.deg2rad(60))
        ang_vel0 = rng.uniform(-2.0, 2.0)
        v_init_max = 3
        v_mag    = np.sqrt(rng.uniform(0, 1)) * v_init_max
        v_dir    = rng.uniform(-np.pi, np.pi)
        vel0     = v_mag * np.exp(1j * v_dir)
        state_matrix[:, 1, :] = vel0
        state_matrix[:, 2, :] = angle0
        state_matrix[:, 3, :] = ang_vel0

    # impulse (wind gust) params — per-tick prob set so ~4 kicks happen per episode on average
    impulse_prob    = 4 * dt / settings['limit']
    impulse_v_sigma = 0.5
    impulse_w_sigma = 0.5

    action_matrix = np.zeros((N, 4, S), dtype=float)

    # simulation state arrays
    toggle = np.zeros((N, S), dtype=bool)
    ticks  = np.zeros((N, S), dtype=int)
    crashed = np.zeros((N, S), dtype=bool) # latched flag: drone has flown out of bounds
    crash_dist = 1.5 * settings['length']  # crash threshold: 1.5x chain length away from target
    particles = ParticlePool(max_particles=4000)
    arangeS  = np.arange(S)
    prev_los = None
    prev_vel = None

    # fitness + descriptor accumulators
    # descriptors: mean gimbal angle, activation variance
    fitness     = np.zeros((N, S))
    track_velo  = np.zeros((N, S))   # raw track sum, sum(dt * track)
    effort_velo = np.zeros((N, S))   # raw effort sum, sum(dt * effort)
    scale_velo  = np.zeros((N, S))   # scaling sum, sum(dt * prox * pretouch_scale)
    sum_acti  = np.zeros((N, 4, S))
    sum_acti2 = np.zeros((N, 4, S))
    mean_gimb = np.zeros(N)
    total_ticks = 0

    sim_time = 0.0
    quit_early = False
    while sim_time < settings['limit']:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                quit_early = True
                break
            if event.type == pg.KEYDOWN and event.key == pg.K_t:
                global RANDOM_SEED_MODE
                RANDOM_SEED_MODE = not RANDOM_SEED_MODE
            if event.type == pg.KEYDOWN and event.key == pg.K_f:
                global FOLLOW_DRONE
                FOLLOW_DRONE = not FOLLOW_DRONE
            if event.type == pg.KEYDOWN and event.key == pg.K_a:
                global SHOW_ARROWS
                SHOW_ARROWS = not SHOW_ARROWS
            if event.type == pg.KEYDOWN and event.key == pg.K_g:
                global SHOW_GHOSTS
                SHOW_GHOSTS = not SHOW_GHOSTS
        if quit_early:
            break

        # alive mask — see sim1.py for rationale. fully-dead drones are skipped from
        # physics + brain to save compute.
        alive_drones = ~crashed.all(axis=1)
        if not alive_drones.any():
            break
        all_alive = bool(alive_drones.all())

        # UPDATE PHYSICS — full batch when nothing's dead, otherwise alive subset only.
        if all_alive:
            state_matrix = physics_update(dt, state_matrix, action_matrix, drone_conf)
        else:
            sub_state = physics_update(dt, state_matrix[alive_drones], action_matrix[alive_drones], drone_conf)
            state_matrix[alive_drones] = sub_state

        # RANDOM IMPULSES (wind gusts) — shared across drones within a seed for fairness.
        # gated by perturbations flag.
        if perturbations:
            mask   = rng.random(S) < impulse_prob
            v_kick = (rng.normal(0, impulse_v_sigma, S) + 1j * rng.normal(0, impulse_v_sigma, S)) * mask
            w_kick = rng.normal(0, impulse_w_sigma, S) * mask
            state_matrix[:, 1, :] += v_kick
            state_matrix[:, 3, :] += w_kick

        # per-(drone, trial) target — clamp ticks to last chain entry per trial
        idx = np.minimum(ticks, tick_pos.shape[1] - 1)           # (N, S)
        target = tick_pos[arangeS, idx]                          # (N, S)

        # MAKE OBSERVATION ARRAY (27 inputs, mirror sim1.py) -------------------------------
        angle        = state_matrix[:, 2, :].real
        delta_world  = target - state_matrix[:, 0, :]
        delta_local  = delta_world * np.exp(-1j * angle)

        # --- target geometry ---
        dist  = np.abs(delta_world)
        los_u = delta_world / (dist + eps)
        los_local = delta_local / (dist + eps)
        if prev_los is None:
            prev_los = los_u
        los_rate = np.tanh((np.angle(los_u / prev_los) / dt - state_matrix[:, 3, :].real) / 3.0)
        prev_los = los_u

        # --- self velocity (local) ---
        vel = state_matrix[:, 1, :] * np.exp(-1j * angle)
        vel_mag = np.abs(vel)
        vel_u   = vel / (vel_mag + eps)

        # --- relative velocity (drone - target, local) ---
        prev_idx       = np.maximum(idx - 1, 0)
        v_target       = (target - tick_pos[arangeS, prev_idx]) / dt
        v_target_local = v_target * np.exp(-1j * angle)
        rel_vel = vel - v_target_local
        relvel_mag = np.abs(rel_vel)
        relvel_u   = rel_vel / (relvel_mag + eps)

        # --- prev-tick net acceleration (world -> local) ---
        vel_world = state_matrix[:, 1, :].copy()
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
        grav_body = (-1j * drone_conf['G']) * np.exp(-1j * angle)
        zem = delta_local - rel_vel * tti_raw + 0.5 * grav_body * tti_raw ** 2
        zev = v_target_local - vel + grav_body * tti_raw
        zem_a = zem / (tti_raw ** 2 + eps)
        zev_a = zev / (tti_raw + eps)
        zem_a = zem_a / (np.abs(zem_a) + eps) * np.tanh(np.abs(zem_a) / max_a)
        zev_a = zev_a / (np.abs(zev_a) + eps) * np.tanh(np.abs(zev_a) / max_a)
        zema_mag = np.abs(zem_a); zema_u = zem_a / (zema_mag + eps)
        zeva_mag = np.abs(zev_a); zeva_u = zev_a / (zeva_mag + eps)

        # --- attitude / actuators ---
        ang_vel   = state_matrix[:, 3, :].real
        t1_ang    = state_matrix[:, 4, :].real
        t2_ang    = state_matrix[:, 5, :].real
        # actual integrated thrust level (not the command); left in [0,1] to match sigmoid output
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
        ], axis=1)  # (N, 27, S)

        # pass alive mask so brain skips fully-dead drones; their outputs hold last tick's.
        if all_alive:
            action_matrix = Brain.forward(obs)  # (N, 4, S)
        else:
            action_matrix = Brain.forward(obs, alive=alive_drones, prev_actions=action_matrix)

        # PROGRESS SIMULATION
        toggle |= dist < 0.5

        # crash detection (latched): drone has flown more than crash_dist away from target.
        # zero crash tolerance: any (drone, seed) pair that ever crashes gets its FULL
        # episode total zeroed at the end of the sim — not just post-crash ticks.
        crashed |= dist > crash_dist

        # FITNESS (track * effort * prox, mirror sim1.py) ---------------------------------
        # SHARED ideal velocity (used by both components)
        v_tgt_par    = (v_target * np.conj(los_u)).real
        safe_v       = 0.8 * np.sqrt(2 * max_a * (dist + eps_d))
        smooth_scale = np.clip(dist / 0.5, 0.0, 1.0)
        safe_v_term  = safe_v * smooth_scale
        ideal_vel    = v_target + safe_v_term * los_u

        # TRACKING component (vratio): achieved velocity vs ideal_vel
        track_err   = np.abs(state_matrix[:, 1, :] - ideal_vel)
        track_scale = np.maximum(np.abs(ideal_vel), floor)
        track = np.clip(1.0 - (track_err / track_scale) ** 2, 0.0, 1.0)

        # EFFORT component (regime-aware projection match) — mirrors sim1.py
        v_free = prev_vel + -9.81j * dt
        # per-tick dv budget scaled by rate-limited throttle delta (mirrors sim1.py)
        dv = max_a * dt * (drone_conf['th_actuation_rate'] * dt)

        err_v    = ideal_vel - v_free
        err_mag  = np.abs(err_v)
        err_unit = err_v / (err_mag + eps)

        actual_dv        = state_matrix[:, 1, :] - v_free
        projection       = (actual_dv * np.conjugate(err_unit)).real
        ideal_projection = np.minimum(err_mag, dv)
        effort_divisor   = np.maximum(ideal_projection, effort_floor)

        prev_vel   = vel_world
        effort_err = projection - ideal_projection
        effort_raw = np.clip(1.0 - (effort_err / effort_divisor) ** 2, 0.0, 1.0)

        # criticality gate (mirrors sim1.py): suppresses effort jiggle during hover.
        criticality = 1.0 - np.exp(-crit_coef * err_mag * err_mag)
        effort      = 1.0 - criticality * (1.0 - effort_raw)

        # FINAL = track * effort * prox * pretouch (matches sim1.py raw scoring)
        x_norm = dist * inv_L
        prox = np.maximum(0.0, 1.0 / (SCALE_K * (x_norm + SCALE_A)) - SCALE_A)
        pretouch_scale = np.where(toggle, 1.0, 1.0)

        score = track * effort * prox * pretouch_scale
        fitness     += dt * score
        track_velo  += dt * track                       # raw track quality
        effort_velo += dt * effort                      # raw effort quality
        scale_velo  += dt * prox * pretouch_scale       # combined scaling

        # descriptors
        normalized = action_matrix / np.array([1, 1, 2, 2]).reshape(1, 4, 1)
        sum_acti  += normalized
        sum_acti2 += normalized ** 2
        mean_gimb += ((np.abs(state_matrix[:, 4, :]) + np.abs(state_matrix[:, 5, :])) / 2).mean(axis=1)

        ticks += toggle
        total_ticks += 1
        sim_time += dt

        # ---- RENDER (trial 0 only) ----
        # Slice down to trial 0 for everything visual. State views are 2D (N, 6) again.
        state_t0  = state_matrix[:, :, 0]
        action_t0 = action_matrix[:, :, 0]
        target_t0 = target[:, 0]

        # camera follows either the highlighted drone or its current target. toggled with F.
        global CAMERA
        if highlight is not None:
            if FOLLOW_DRONE:
                CAMERA = complex(state_t0[highlight, 0])
            else:
                CAMERA = complex(target_t0[highlight])

        # particles + draw
        spawn_thruster_particles(particles, state_t0, action_t0, drone_conf, [highlight])
        particles.update(dt)

        screen.fill((20, 20, 20))
        particles.draw(screen)

        # draw all S target chains + approach lines: trial 0 full color, others very faint
        chain_layer = pg.Surface((SCREEN_W, SCREEN_H), pg.SRCALPHA)
        stride = max(1, tick_pos.shape[1] // 200)
        origin_px = world_to_screen(0j)
        for s in range(S):
            if s == 0:
                chain_color    = (60, 100, 60, 255)
                approach_color = (200, 60, 60, 255)
            else:
                chain_color    = (60, 100, 60, 60)
                approach_color = (200, 60, 60, 60)
            # approach line: origin → wp1 for this trial
            pg.draw.line(chain_layer, approach_color, origin_px, world_to_screen(tick_pos[s, 0]), 1)
            # path through the chain
            pts = [world_to_screen(p) for p in tick_pos[s, ::stride]]
            if len(pts) > 1:
                pg.draw.lines(chain_layer, chain_color, False, pts, 1)
        screen.blit(chain_layer, (0, 0))

        # non-featured: simple dots (trial 0 positions only). gated by SHOW_GHOSTS (G).
        if SHOW_GHOSTS:
            dot_layer = pg.Surface((SCREEN_W, SCREEN_H), pg.SRCALPHA)
            for i in range(N):
                if i in featured:
                    continue
                sx, sy = world_to_screen(state_t0[i, 0])
                pg.draw.circle(dot_layer, (180, 180, 180, 70), (sx, sy), 3)
            screen.blit(dot_layer, (0, 0))

        # featured (incl. highlight): low-alpha full sprites — trial 0 only.
        # gated by SHOW_GHOSTS (G) along with the dots above and the yellow ring below.
        if SHOW_GHOSTS:
            for i in featured:
                if i == highlight:
                    continue
                draw_drone(screen, state_t0[i], drone_surf, thruster_surf, drone_conf, alpha=140)
        if highlight is not None:
            if SHOW_GHOSTS:
                hx, hy = world_to_screen(state_t0[highlight, 0])
                pg.draw.circle(screen, (255, 220, 60), (hx, hy), 18, 2)
            draw_drone(screen, state_t0[highlight], drone_surf, thruster_surf, drone_conf, alpha=255)
            tx, ty = world_to_screen(target_t0[highlight])
            pg.draw.circle(screen, (100, 230, 100), (tx, ty), 3)

            # scoring vectors for the highlighted drone (trial 0) — gated by SHOW_ARROWS (A).
            if SHOW_ARROWS:
                base_pos = state_t0[highlight, 0]
                cur_v = state_matrix[highlight, 1, 0]
                VS  = 0.4   # shared velocity scale
                AMP = 5.0   # amplify the tiny per-tick thrust impulses so they're visible
                draw_vector(screen, base_pos, ideal_vel[highlight, 0], (230, 90, 220), scale=VS)  # track target
                draw_vector(screen, base_pos, cur_v,                   (80, 200, 255), scale=VS)  # current v

                # both impulse arrows rooted at the v_free tip — directly comparable.
                # ORANGE = optimal thrust impulse: err_unit * ideal_projection (= min(|err_v|, dv))
                # RED    = actual_dv produced this tick. red overlapping orange = perfect effort.
                vf   = v_free[highlight, 0]
                root = base_pos + vf * VS
                ideal_dv = err_unit[highlight, 0] * ideal_projection[highlight, 0] * AMP
                draw_vector(screen, root, ideal_dv, (255, 170, 40), scale=VS)

                doing = actual_dv[highlight, 0] * AMP
                draw_vector(screen, root, doing, (255, 80, 80), scale=VS)

                screen.blit(font.render("current v", True, (80, 200, 255)), (10, 40))
                screen.blit(font.render("ideal dv (err_unit * ideal_proj)", True, (255, 170, 40)), (10, 64))
                screen.blit(font.render("actual dv this tick", True, (255, 80, 80)), (10, 88))
                screen.blit(font.render("ideal v (track target)", True, (230, 90, 220)), (10, 112))

        fps = clock.get_fps()
        # tooltips describe what pressing the key would do FROM the current mode.
        seed_label = ("random",  "T: replay current") if RANDOM_SEED_MODE else ("current", "T: randomize")
        cam_label  = ("drone",   "F: follow target")  if FOLLOW_DRONE    else ("target",  "F: follow drone")
        arrow_label= ("on",      "A: hide arrows")    if SHOW_ARROWS     else ("off",     "A: show arrows")
        ghost_label= ("on",      "G: hide ghosts")    if SHOW_GHOSTS     else ("off",     "G: show ghosts")
        screen.blit(font.render(f"FPS: {fps:.0f}  Drones: {N}  t={sim_time:.1f}/{settings['limit']:.1f}  seed: {seed_label[0]} [{seed_label[1]}]  cam: {cam_label[0]} [{cam_label[1]}]  arrows: {arrow_label[0]} [{arrow_label[1]}]  ghosts: {ghost_label[0]} [{ghost_label[1]}]", True, (150, 150, 150)), (10, 10))
        # highlighted drone's running fitness components (mean over seeds, raw quality)
        hi_track  = float(track_velo [highlight].mean())
        hi_effort = float(effort_velo[highlight].mean())
        hi_scale  = float(scale_velo [highlight].mean())
        hi_fit    = float(fitness    [highlight].mean())
        screen.blit(font.render(f"track: {hi_track:.2f}  effort: {hi_effort:.2f}  scale: {hi_scale:.2f}  fitness: {hi_fit:.2f}", True, (180, 180, 180)), (10, 136))

        pg.display.flip()
        clock.tick(max(1, int(60 / PLAYBACK_SLOWDOWN)))

    pg.quit()

    # zero crash tolerance: any (drone, seed) pair that ever crashed gets its full
    # episode contribution wiped — fitness, track, effort, scale all set to 0.
    survived = (~crashed).astype(np.float32)
    fitness     *= survived
    track_velo  *= survived
    effort_velo *= survived
    scale_velo  *= survived

    if total_ticks > 0:
        var_acti  = (sum_acti2 / total_ticks) - (sum_acti / total_ticks) ** 2  # (N, 4, S)
        var_acti  = var_acti.mean(axis=(1, 2))                                  # (N,)
        mean_gimb = mean_gimb / total_ticks                                     # (N,)
    else:
        var_acti = np.zeros(N)

    if update_fitness:
        # SE-penalized fitness (mean - std/sqrt(S)) — matches sim1.sim's selection metric.
        # note: visual sim uses S=4 so SE is a coarser estimate than training's S=32.
        per_seed_mean = fitness.mean(axis=1)
        per_seed_std  = fitness.std (axis=1)
        per_drone_fit = per_seed_mean - per_seed_std / np.sqrt(S)
        for i, ind in enumerate(individuals):
            ind.fitness = float(per_drone_fit[i])
            ind.descriptors = {'mean_gimb': mean_gimb[i], 'var_action': var_acti[i]}

    return {'fit_mean': fitness.mean(), 'fit_max': fitness.max()}


if __name__ == "__main__":
    import os
    from modules.evo_alg.mapElites import load_alg

    save_path = os.path.join('data', 'MAP_Checkpoint.pkl')
    alg, settings, seed = load_alg(save_path)

    # 3x3 coarse bin → best elite per bin (featured, full sprite),
    # everything else → ghost line art.
    grid = alg.archive.indv
    R, C = grid.shape
    K = 3
    featured_set = []
    seen = set()
    for i in range(K):
        for j in range(K):
            block = grid[i*R//K:(i+1)*R//K, j*C//K:(j+1)*C//K]
            cands = [x for x in block.flat if x is not None]
            if cands:
                best = max(cands, key=lambda x: x.fitness)
                featured_set.append(best)
                seen.add(id(best))

    rest = [x for x in grid.flat if x is not None and id(x) not in seen]
    elites = featured_set + rest
    featured_idx = list(range(len(featured_set)))

    print(f"loaded {len(elites)} elites ({len(featured_set)} featured) from {save_path} (gen {alg.gen})")
    print(f"settings: {settings}")
    try:
        # mode controlled by RANDOM_SEED_MODE (toggled in-visual with the T key).
        # False  -> replay the checkpoint seed every loop (same scenario, same drones)
        # True   -> fresh random seed every loop (new scenarios)
        while True:
            playback_seed = None if RANDOM_SEED_MODE else seed
            sim(elites, settings, seed=playback_seed, featured=featured_idx, update_fitness=False)
    except KeyboardInterrupt:
        pass
