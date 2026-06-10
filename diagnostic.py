from modules.evo_alg.mapElites import load_alg
from modules.simulation.sim1 import sim
import numpy as np
import matplotlib.pyplot as plt
import os


# 40 equal bins over [0,1], final [0.975,1.0] bar split at 0.99 so the
# saturation spike (signal >= 0.99) lands in its own bin. x-range stays [0,1].
HIST_BINS = np.sort(np.append(np.linspace(0, 1, 41), 0.99))

save_path = os.path.join('data', 'MAP_Checkpoint.pkl')


def pick_quartile_elites(archive, per_quartile: int = 5):
    # Sample `per_quartile` evenly-spaced drones from each fitness quartile of the
    # archive (sorted ascending by fitness). Captures real best-vs-worst spread
    # across the trained population — not just spread among top elites.
    drones = [x for x in archive.indv.flat if x is not None and x.fitness is not None]
    if not drones:
        return []
    drones.sort(key=lambda x: x.fitness)   # ascending
    n = len(drones)
    picks = []
    for q in range(4):
        start = q * n // 4
        end   = (q + 1) * n // 4
        qsize = end - start
        if qsize == 0:
            continue
        if qsize <= per_quartile:
            picks.extend(drones[start:end])
        else:
            idxs = np.linspace(start, end - 1, per_quartile).astype(int)
            picks.extend([drones[int(i)] for i in idxs])
    return picks


if __name__ == "__main__":
    alg, settings, seed = load_alg(save_path)
    print(f"loaded checkpoint (gen {alg.gen}) — settings={settings} seed={seed}")

    elites = pick_quartile_elites(alg.archive, per_quartile=5)
    print(f"picked {len(elites)} drones across 4 fitness quartiles "
          f"(fitness range: {elites[0].fitness:.3f} → {elites[-1].fitness:.3f})")

    # run the sim with per-tick logging on the saved seed → same regime as training
    _, stats = sim(elites, settings, seed=seed, log_per_tick=True)

    tick_fit             = stats['tick_fit']               # (N, S, T)
    tick_track           = stats['tick_track']             # ema-smoothed (K=4) track quality
    tick_effort          = stats['tick_effort']            # ema-smoothed (K=16) effort quality (no gate)
    tick_scale           = stats['tick_scale']
    tick_track_raw       = stats['tick_track_raw']         # raw, pre-EMA
    tick_effort_raw      = stats['tick_effort_raw']        # raw, pre-EMA
    tick_effort_weighted = stats['tick_effort_weighted']   # criticality removed; == tick_effort_raw
    tick_discount        = stats['tick_discount']          # time-pressure decay, gamma**ticks_since_touch
    tick_rel             = stats['tick_rel']               # drone pos in target frame (complex)
    # distance to target, logged at collection. floored at LOG_EPS so hovering right
    # on the target stays finite. all distance-based plots use this log scale — the
    # raw range spans orders of magnitude (cm hover vs far approach), so log spreads
    # the interesting near-target behavior instead of crushing it against zero.
    LOG_EPS              = 1e-2
    tick_dist            = np.log10(np.maximum(np.abs(tick_rel), LOG_EPS))  # (N, S, T) log10 distance
    dt                   = stats['dt']
    N, S, T              = tick_fit.shape

    # best drone selection — uses SE-penalized fitness (matches sim1.sim's selection metric).
    # episode totals are raw sums; the per-drone aggregation across seeds is what shifts.
    drone_fit_ema  = tick_fit.sum(axis=2)                            # (N, S) raw episode totals
    drone_mean_ema = drone_fit_ema.mean(axis=1)                      # (N,) raw mean
    drone_se_fit   = drone_mean_ema - drone_fit_ema.std(axis=1) / np.sqrt(S)
    best_d         = int(drone_se_fit.argmax())
    best_s         = int(drone_fit_ema[best_d].argmax())             # best seed within best drone, by raw

    ind = elites[best_d]
    print(f"best drone idx={best_d} (tag={ind.tag}) seed={best_s} "
          f"ema_fit_raw={drone_fit_ema[best_d, best_s]:.3f} "
          f"se_fit={drone_se_fit[best_d]:.3f} "
          f"descr={ind.descriptors}")

    # extract signals for the chosen (drone, seed). EMA'd values are what training saw:
    # tracking is EMA'd at K=4, effort at K=16. there is no criticality gate (removed) —
    # effort is the raw projection-match score straight into its EMA.
    fit_signal             = tick_fit            [best_d, best_s]      # dt-scaled reward increments
    scale_signal    = tick_scale   [best_d, best_s] / dt
    track_signal    = tick_track   [best_d, best_s] / dt   # ema'd track quality
    effort_signal   = tick_effort  [best_d, best_s] / dt   # ema'd effort quality
    discount_signal = tick_discount[best_d, best_s]        # already a [0,1] factor, not dt-scaled

    # nested-headroom gating (keep LAMBDA/MU in sync with sim1.py). pre-gate = the
    # component's raw quality in [0,1] (what it achieved); post-gate = its actual
    # additive contribution to the per-tick score after the headroom leverage:
    #   score = prox + LAMBDA(1-prox)*track + LAMBDA*MU(1-prox)(1-track)*effort
    LAMBDA_BRIDGE = 0.95
    MU_BRIDGE     = 0.30
    track_pre   = track_signal
    track_post  = LAMBDA_BRIDGE * (1.0 - scale_signal) * track_signal
    effort_pre  = effort_signal
    effort_post = LAMBDA_BRIDGE * MU_BRIDGE * (1.0 - scale_signal) * (1.0 - track_signal) * effort_signal

    cum_fit  = np.cumsum(fit_signal)
    time     = np.arange(T) * dt
    rel_best = tick_rel[best_d, best_s]                  # drone pos in target frame (complex)
    fit_pt   = fit_signal / dt                           # per-tick fitness (for trajectory color)

    # 2x3 layout: four signal time-series on the left/middle, target-centered
    # trajectory map spanning the full right column (square, gets the most room).
    fig = plt.figure(figsize=(18, 9))
    gs  = fig.add_gridspec(2, 3)
    ax_cum    = fig.add_subplot(gs[0, 0])
    ax_track  = fig.add_subplot(gs[0, 1])
    ax_effort = fig.add_subplot(gs[1, 0])
    ax_scale  = fig.add_subplot(gs[1, 1])
    ax_traj   = fig.add_subplot(gs[:, 2])

    ax = ax_cum
    ax.plot(time, cum_fit, color='C0')
    ax.set_title(f'cumulative fitness — drone {best_d}, seed {best_s} (final={cum_fit[-1]:.3f})')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('cum fitness')
    ax.grid(alpha=0.3)

    def plot_pre_post(ax, pre, post, color, title, ylabel):
        ax.plot(time, pre,  color='gray', alpha=0.45, linewidth=0.9, label='pre-gate (quality)')
        ax.plot(time, post, color=color,  linewidth=1.3,             label='post-gate (score contribution)')
        ax.set_title(title)
        ax.set_xlabel('time (s)')
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)

    plot_pre_post(ax_track,  track_pre,  track_post,  'C1',
                  'track — quality vs score contribution', 'track')
    plot_pre_post(ax_effort, effort_pre, effort_post, 'C2',
                  'effort — quality vs score contribution', 'effort')

    # scale (proximity, no EMA) + time-pressure discount overlaid. both in [0,1]:
    # where discount dips below 1 while scale is still high = "near but not touching"
    # (drone lingering inside the prox band but never dipping under the 0.5 touch radius).
    ax = ax_scale
    ax.plot(time, scale_signal,    color='C3', linewidth=1.0, label='scale (prox)')
    ax.plot(time, discount_signal, color='gray', linewidth=1.0, alpha=0.7,
            linestyle='--', label='time discount')
    ax.set_title('scale + time-pressure discount')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('scale / discount')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)

    # target-centered trajectory map: target fixed at origin (red +), drone's path
    # plotted in the target frame. each point colored by the per-tick fitness it
    # earned there (LINEAR color scale — a log scale would hide unsaturation and
    # other low-fitness issues). faint line shows path order; star marks the start.
    ax = ax_traj
    ax.plot(rel_best.real, rel_best.imag, color='gray', alpha=0.25, linewidth=0.6, zorder=1)
    sc = ax.scatter(rel_best.real, rel_best.imag, c=fit_pt, cmap='viridis',
                    s=12, zorder=2, vmin=0.0, vmax=max(float(fit_pt.max()), 1e-6))
    ax.scatter([0], [0], marker='+', color='red', s=200, linewidths=2.5, zorder=4, label='target')
    ax.scatter([rel_best.real[0]], [rel_best.imag[0]], marker='*', color='black',
               s=120, zorder=5, label='start')
    fig.colorbar(sc, ax=ax, label='fitness at point (per tick)', fraction=0.046, pad=0.04)
    ax.set_title(f'trajectory in target frame — drone {best_d}, seed {best_s}\n'
                 f'mean log₁₀ dist={tick_dist[best_d, best_s].mean():.2f}')
    ax.set_xlabel('x − target_x')
    ax.set_ylabel('y − target_y')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(f'diagnostic — best across quartile sampling (gen {alg.gen}, seed {seed})')
    fig.tight_layout()
    fig.savefig('diagnostic.png', dpi=150)

    # =========================================================================
    # signal self-analysis — same 4-panel diagnostic for effort, track, scale.
    # comparison is best drone (Q4 top) vs worst-of-Q3 (~median). picking
    # worst-of-Q3 instead of absolute worst gives a "great vs mediocre"
    # discrimination test — more informative for trained-pop signal design
    # than "great vs broken" which the signal already trivially separates.
    # Q3 occupies positions [2*per_q, 3*per_q) in the ascending-sorted elites
    # list from pick_quartile_elites. its bottom (= ~50th percentile drone)
    # is the first drone of that range.
    #   (0,0) time-series overlay        (0,2) CCDF high-score tail
    #   (0,1) value histogram            (1,1) consecutive-streak histogram
    #   (1,0) windowed scatter vs a companion quality signal (non-circular check)
    #   (1,2) within-episode signal vs distance (windowed, best & Q3) — does this
    #         signal stay high while the drone is actually close to the target over
    #         the episode? distance on x (lower = closer = better); a useful signal
    #         sits HIGH at LOW distance -> upper-left cluster. single-drone traces,
    #         same style as the other per-signal panels (not a per-drone·seed view).
    # =========================================================================
    PER_QUARTILE = 5  # matches pick_quartile_elites call above
    q3_start = 1 * PER_QUARTILE
    if q3_start < N:
        worst_d = q3_start                            # bottom-of-Q3 = ~median drone
    else:
        worst_d = int(drone_se_fit.argmin())          # fallback if archive too small
    best_fit_ema  = float(drone_fit_ema[best_d,  best_s])
    worst_fit_ema = float(drone_fit_ema[worst_d, best_s])

    def ccdf(sig: np.ndarray, grid: np.ndarray) -> np.ndarray:
        # complementary CDF: P(X > x) for each x in grid. survival function.
        # at the high end (x→1) the better controller stays higher + flatter,
        # i.e. it spends a larger fraction of ticks above any near-1 threshold.
        s = np.sort(sig)
        counts = len(s) - np.searchsorted(s, grid, side='right')   # #{ X > x }
        return counts / max(len(s), 1)

    def streak_lengths(sig: np.ndarray, thresh: float) -> np.ndarray:
        # run lengths (in ticks) of consecutive samples with sig > thresh.
        # best controller locks on: a few very long runs. mediocre one flickers:
        # many short runs. returns empty array if it never crosses thresh.
        mask = (sig > thresh).astype(np.int8)
        if mask.sum() == 0:
            return np.array([], dtype=np.int64)
        padded = np.concatenate(([0], mask, [0]))
        diffs  = np.diff(padded)
        starts = np.where(diffs ==  1)[0]
        ends   = np.where(diffs == -1)[0]
        return (ends - starts).astype(np.int64)

    def signal_diagnostic(name: str, best_sig: np.ndarray, worst_sig: np.ndarray,
                          companion_sig: np.ndarray, companion_name: str,
                          color_best: str, color_worst: str, save_name: str):
        W_corr = 32
        n_win  = T - W_corr
        if n_win > 0:
            sig_w = np.array([best_sig     [t:t + W_corr].mean() for t in range(n_win)], dtype=np.float32)
            com_w = np.array([companion_sig[t:t + W_corr].mean() for t in range(n_win)], dtype=np.float32)
            if sig_w.std() > 1e-12 and com_w.std() > 1e-12:
                corr_r = float(np.corrcoef(sig_w, com_w)[0, 1])
            else:
                corr_r = float('nan')
        else:
            sig_w = com_w = np.array([], dtype=np.float32)
            corr_r = float('nan')

        f, ax = plt.subplots(2, 3, figsize=(19, 9))

        a = ax[0, 0]
        a.plot(time, best_sig,  color=color_best,  alpha=0.8, linewidth=1.0, label=f'best drone {best_d}')
        a.plot(time, worst_sig, color=color_worst, alpha=0.8, linewidth=1.0, label=f'Q3 drone {worst_d}')
        a.set_title(f'{name} — best (fit={best_fit_ema:.2f}) vs Q3 (fit={worst_fit_ema:.2f})')
        a.set_xlabel('time (s)')
        a.set_ylabel(name)
        a.set_ylim(-0.05, 1.05)
        a.grid(alpha=0.3)
        a.legend(loc='lower right', fontsize=8)

        a = ax[0, 1]
        bins = HIST_BINS
        a.hist(best_sig,  bins=bins, color=color_best,  alpha=0.5,
               label=f'best  μ={best_sig.mean():.3f}  σ={best_sig.std():.3f}')
        a.hist(worst_sig, bins=bins, color=color_worst, alpha=0.5,
               label=f'Q3    μ={worst_sig.mean():.3f}  σ={worst_sig.std():.3f}')
        a.set_title(f'{name} distribution')
        a.set_xlabel(f'{name} value')
        a.set_ylabel('tick count')
        a.grid(alpha=0.3)
        a.legend(loc='upper center', fontsize=8)

        a = ax[1, 0]
        a.scatter(sig_w, com_w, color=color_best, alpha=0.3, s=8)
        a.set_title(f'windowed mean (W={W_corr}): {name} vs {companion_name} — r={corr_r:.3f}')
        a.set_xlabel(f'mean({name})')
        a.set_ylabel(f'mean({companion_name})')
        a.set_xlim(-0.05, 1.05)
        a.set_ylim(-0.05, 1.05)
        a.grid(alpha=0.3)

        # (0,2) CCDF over the full [0, 1] range. P(X > x): fraction of ticks held
        # above threshold x. better controller's curve stays higher + flatter toward
        # 1 — it maintains near-perfect score for a larger share of ticks. dashed
        # marker at 0.95 keeps the high-score tail readable.
        a = ax[0, 2]
        grid = np.linspace(0.0, 1.0, 501)
        ccdf_best  = ccdf(best_sig,  grid)
        ccdf_worst = ccdf(worst_sig, grid)
        a.plot(grid, ccdf_best,  color=color_best,  linewidth=1.3,
               label=f'best  P(>0.95)={(best_sig  > 0.95).mean():.2f}')
        a.plot(grid, ccdf_worst, color=color_worst, linewidth=1.3,
               label=f'Q3    P(>0.95)={(worst_sig > 0.95).mean():.2f}')
        a.axvline(0.95, color='gray', linewidth=0.5, linestyle='--')
        a.set_title(f'{name} CCDF — P(X > x)')
        a.set_xlabel(f'threshold x  ({name} value)')
        a.set_ylabel('P(X > x)')
        a.set_xlim(-0.02, 1.02)
        a.set_ylim(-0.02, 1.02)
        a.grid(alpha=0.3)
        a.legend(loc='upper right', fontsize=8)

        # (1,1) streak-length survival: P(run length >= L), where a run is a maximal
        # block of consecutive ticks with score > 0.95. survival handles the heavy
        # tail a histogram can't: best controller holds a few very long runs so its
        # curve stays high far to the right; a flickering drone collapses to zero
        # within a few ticks (only brief flashes, never a sustained lock).
        a = ax[1, 1]
        STREAK_THRESH = 0.95
        sk_best  = streak_lengths(best_sig,  STREAK_THRESH)
        sk_worst = streak_lengths(worst_sig, STREAK_THRESH)
        max_run  = int(max(sk_best.max()  if sk_best.size  else 0,
                           sk_worst.max() if sk_worst.size else 0, 1))
        Lgrid = np.arange(1, max_run + 1)
        surv_best  = np.array([float((sk_best  >= L).mean()) if sk_best.size  else 0.0 for L in Lgrid])
        surv_worst = np.array([float((sk_worst >= L).mean()) if sk_worst.size else 0.0 for L in Lgrid])
        a.step(Lgrid, surv_best,  where='post', color=color_best,  linewidth=1.4,
               label=f'best  n={sk_best.size}  max={int(sk_best.max())  if sk_best.size  else 0}t')
        a.step(Lgrid, surv_worst, where='post', color=color_worst, linewidth=1.4,
               label=f'Q3    n={sk_worst.size}  max={int(sk_worst.max()) if sk_worst.size else 0}t')
        a.set_title(f'{name} streak survival > {STREAK_THRESH} (max {max_run}t = {max_run*dt:.1f}s)')
        a.set_xlabel('run length L (consecutive ticks)')
        a.set_ylabel('P(run length ≥ L)')
        a.set_ylim(-0.02, 1.02)
        a.grid(alpha=0.3)
        a.legend(loc='upper right', fontsize=8)

        # (1,2) within-episode signal vs distance — raw per-tick points for best & Q3
        # over this one episode. log₁₀ distance on the X axis, INVERTED so right =
        # closer to target. a useful signal sits HIGH on the right (high score while
        # close), so good behavior clusters toward the upper-RIGHT. r is best-drone's
        # signal↔log-distance correlation.
        a = ax[1, 2]
        best_dist  = tick_dist[best_d,  best_s]   # log10 distance, per tick
        worst_dist = tick_dist[worst_d, best_s]
        a.scatter(best_dist,  best_sig,  color=color_best,  alpha=0.4, s=8, label=f'best drone {best_d}')
        a.scatter(worst_dist, worst_sig, color=color_worst, alpha=0.4, s=8, label=f'Q3 drone {worst_d}')
        if best_sig.size > 1 and best_dist.std() > 1e-12 and best_sig.std() > 1e-12:
            r_sd = float(np.corrcoef(best_dist, best_sig)[0, 1])
        else:
            r_sd = float('nan')
        a.set_title(f'{name} vs distance (per tick) — best r={r_sd:+.2f}\n'
                    f'(log₁₀ dist on x, right = closer; useful signal sits high on the right)')
        a.set_xlabel('log₁₀ distance to target')
        a.set_ylabel(name)
        a.set_ylim(-0.05, 1.05)
        a.invert_xaxis()   # right = decreasing distance (closer to target)
        a.grid(alpha=0.3)
        a.legend(loc='upper left', fontsize=8)

        f.suptitle(f'{name} signal diagnostic — same seed {best_s}')
        f.tight_layout()
        f.savefig(save_name, dpi=150)
        return f

    # extract per-(drone, seed) signals for the best vs Q3 comparison
    best_eff  = tick_effort[best_d,  best_s] / dt
    worst_eff = tick_effort[worst_d, best_s] / dt
    best_trk  = tick_track [best_d,  best_s] / dt
    worst_trk = tick_track [worst_d, best_s] / dt
    best_scl  = tick_scale [best_d,  best_s] / dt
    worst_scl = tick_scale [worst_d, best_s] / dt

    # per-(drone, seed) episode means: mean signal and mean distance, flattened to
    # one entry per (drone, seed). drives the (1,2) "signal -> goal" panel.
    ep_dist_mean   = tick_dist.mean(axis=2).flatten()              # (N*S,) lower is better
    ep_track_mean  = (tick_track  / dt).mean(axis=2).flatten()
    ep_effort_mean = (tick_effort / dt).mean(axis=2).flatten()
    ep_scale_mean  = (tick_scale  / dt).mean(axis=2).flatten()
    # correlation of each signal (and each unique signal-product pair) to mean
    # distance, across all (drone, seed) episode means — i.e. "does a drone scoring
    # higher on this signal actually end up closer to the target". distance is the
    # real objective and LOWER is better, so a useful signal has NEGATIVE r. pairs
    # use the product since fitness is multiplicative (track * effort * scale).
    def corr_to_dist(x: np.ndarray) -> float:
        if x.size > 1 and x.std() > 1e-12 and ep_dist_mean.std() > 1e-12:
            return float(np.corrcoef(x, ep_dist_mean)[0, 1])
        return float('nan')

    print("signal → mean distance correlation (per drone·seed episode means; negative = good):")
    print(f"  track            r={corr_to_dist(ep_track_mean):+.3f}")
    print(f"  effort           r={corr_to_dist(ep_effort_mean):+.3f}")
    print(f"  scale            r={corr_to_dist(ep_scale_mean):+.3f}")
    print(f"  track*effort     r={corr_to_dist(ep_track_mean * ep_effort_mean):+.3f}")
    print(f"  track*scale      r={corr_to_dist(ep_track_mean * ep_scale_mean):+.3f}")
    print(f"  effort*scale     r={corr_to_dist(ep_effort_mean * ep_scale_mean):+.3f}")

    # one figure per component. companion is a different quality signal for
    # non-circular correlation check. effort↔track and track↔scale are the
    # natural pairings (independent physical levels: impulse, velocity, position).
    signal_diagnostic('effort', best_eff, worst_eff, best_trk, 'track',
                      'C2', 'C3', 'diagnostic_effort.png')
    signal_diagnostic('track',  best_trk, worst_trk, best_scl, 'scale',
                      'C1', 'C5', 'diagnostic_track.png')
    signal_diagnostic('scale',  best_scl, worst_scl, best_trk, 'track',
                      'C3', 'C4', 'diagnostic_scale.png')

    # =========================================================================
    # cross-signal pair plot — best drone, best seed.
    # the 4th variable is DISTANCE-to-target (the real objective), replacing the
    # old fit/tick — comparing signals against fitness was circular (they ARE
    # fitness). now every signal can be read against the actual goal.
    # diagonal:        per-signal histogram
    # lower triangle:  windowed-mean scatter (single drone, within-episode)
    # upper triangle:  per-(drone, seed) episode means, colored by mean distance
    #                  (viridis_r -> brighter = closer = better)
    # column names labeled along the top row, row names along the left column.
    # =========================================================================
    dist_best = tick_dist[best_d, best_s]                  # per-tick log10 distance, best drone
    # shared log-distance axis range covering both within-episode (per-tick) and
    # across-drone (episode-mean) distance data so nothing clips. log distance can be
    # negative (sub-unit hover), so span [min, max] with a small pad.
    DIST_LO = float(min(dist_best.min(), ep_dist_mean.min()))
    DIST_HI = float(max(dist_best.max(), ep_dist_mean.max()))
    _dpad   = 0.05 * (DIST_HI - DIST_LO) + 1e-6

    UNIT = (-0.05, 1.05)                                    # [0,1] signal range
    DLIM = (DIST_LO - _dpad, DIST_HI + _dpad)               # log-distance range
    DIST_BINS = np.linspace(DIST_LO, DIST_HI, 41)           # diagonal hist bins for distance
    # (name, per-tick signal, color, axis-range, is_distance)
    pair_signals = [
        ('track',    best_trk,  'C1', UNIT, False),
        ('effort',   best_eff,  'C2', UNIT, False),
        ('scale',    best_scl,  'C3', UNIT, False),
        ('log dist', dist_best, 'C0', DLIM, True),
    ]
    n_sig = len(pair_signals)

    # windowed means for the lower-triangle scatter — matches signal_diagnostic's
    # W=32-tick scatter so the same plot in both figures shows the same data.
    W_pair = 32
    def windowed_mean(x: np.ndarray, W: int = W_pair) -> np.ndarray:
        n = x.size - W
        return np.array([x[t:t + W].mean() for t in range(n)], dtype=np.float32) if n > 0 else x

    pair_windowed = [windowed_mean(sig) for _, sig, _, _, _ in pair_signals]

    # per-(drone, seed) episode means for the upper-triangle scatter — one point per
    # (drone, seed). reveals whether (sig_i, sig_j) plane *separates* good drones from
    # bad ones, which is the actual selection question. complements the lower triangle
    # (which is single-drone within-episode behavior).
    ep_arrays = {
        'track':    ep_track_mean,
        'effort':   ep_effort_mean,
        'scale':    ep_scale_mean,
        'log dist': ep_dist_mean,
    }
    ep_drone_idx  = np.broadcast_to(np.arange(N)[:, None], (N, S)).flatten()
    drone_mean_dist = tick_dist.mean(axis=(1, 2))            # (N,) mean log distance per drone
    ep_drone_dist = drone_mean_dist[ep_drone_idx]           # color = mean log distance (lower=better)

    def correlation_ratio(values: np.ndarray, group_idx: np.ndarray) -> float:
        # η² (eta squared) = between-group variance / total variance.
        # captures nonlinear / categorical association — high η² with low r means
        # the signal discriminates groups (drones) but not via a linear trend.
        overall = float(values.mean())
        ss_total = float(((values - overall) ** 2).sum())
        if ss_total <= 0:
            return 0.0
        ss_between = 0.0
        for g in np.unique(group_idx):
            mask = group_idx == g
            n_g  = int(mask.sum())
            if n_g > 0:
                ss_between += n_g * (float(values[mask].mean()) - overall) ** 2
        return float(ss_between / ss_total)

    fig3, axes3 = plt.subplots(n_sig, n_sig, figsize=(12, 12))
    ep_sc = None
    for i, (name_i, sig_i, color_i, lim_i, is_dist_i) in enumerate(pair_signals):
        for j, (name_j, sig_j, color_j, lim_j, is_dist_j) in enumerate(pair_signals):
            a = axes3[i, j]
            if i == j:
                # DIAGONAL: histogram (distance gets its own bins; others use [0,1] bins)
                bins_i = DIST_BINS if is_dist_i else HIST_BINS
                a.hist(sig_i, bins=bins_i, color=color_i, alpha=0.7)
                a.set_xlim(*lim_i)
                a.set_yticks([])
            elif i > j:
                # LOWER TRIANGLE: scatter of windowed means (matches signal_diagnostic).
                win_i = pair_windowed[i]
                win_j = pair_windowed[j]
                a.scatter(win_j, win_i, color='gray', alpha=0.4, s=4)
                r = float(np.corrcoef(win_j, win_i)[0, 1]) if win_i.size > 1 else float('nan')
                a.text(0.03, 0.97, f'r={r:+.2f} (W={W_pair})', transform=a.transAxes,
                       va='top', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                a.set_xlim(*lim_j)
                a.set_ylim(*lim_i)
            else:
                # UPPER TRIANGLE: per-(drone, seed) episode means, colored by the drone's
                # mean distance to target (viridis_r -> brighter = closer = better).
                ep_x = ep_arrays[name_j]
                ep_y = ep_arrays[name_i]
                ep_sc = a.scatter(ep_x, ep_y, c=ep_drone_dist, cmap='viridis_r',
                                  alpha=0.7, s=14, edgecolors='none')
                if ep_x.size > 1:
                    r_ep = float(np.corrcoef(ep_x, ep_y)[0, 1])
                else:
                    r_ep = float('nan')
                # η² of the row signal grouped by drone — captures nonlinear
                # group separation that pearson r misses. high η² with low |r|
                # means drones cluster distinctly in the row dimension even
                # without a monotonic relationship to the column dimension.
                eta2 = correlation_ratio(ep_y, ep_drone_idx)
                a.text(0.03, 0.97, f'r={r_ep:+.2f}  η²={eta2:.2f}',
                       transform=a.transAxes, va='top', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))
                a.set_xlim(*lim_j)
                a.set_ylim(*lim_i)

            # flip distance axes so the GOOD end (distance -> 0) lands where the other
            # signals' good end (value -> 1) is: top / right. keeps "good" in the same
            # corner of every panel. diagonal x is signal i; off-diagonal x is signal j.
            if i == j:
                if is_dist_i:
                    a.invert_xaxis()
            else:
                if is_dist_j:
                    a.invert_xaxis()
                if is_dist_i:
                    a.invert_yaxis()

            # column header on the top row (read along the top to identify columns)
            if i == 0:
                a.set_title(name_j, fontsize=12, pad=8, weight='bold')
            # row label on the left column (read down the side to identify rows)
            if j == 0:
                a.set_ylabel(name_i, fontsize=12, weight='bold')

            # hide inner tick labels to reduce clutter; outer edges keep them
            if i != n_sig - 1:
                a.tick_params(labelbottom=False)
            if j != 0:
                a.tick_params(labelleft=False)

    fig3.suptitle(f'cross-signal pair plot — best drone {best_d}, seed {best_s}\n'
                  f'lower = single-drone windowed scatter (within-episode),  '
                  f'upper = per-(drone, seed) episode means (across drones, color = mean distance, brighter=closer)',
                  fontsize=10)
    fig3.tight_layout()
    if ep_sc is not None:
        fig3.colorbar(ep_sc, ax=axes3, label='mean log₁₀ distance to target (lower = better)',
                      fraction=0.025, pad=0.01)
    fig3.savefig('diagnostic_pairs.png', dpi=150)

    plt.show()
