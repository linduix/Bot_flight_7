from modules.evo_alg.mapElites import load_alg
from modules.simulation.sim1 import sim
import numpy as np
import matplotlib.pyplot as plt
import os


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
    tick_track           = stats['tick_track']             # ema-smoothed
    tick_effort          = stats['tick_effort']            # ema-smoothed (post-criticality)
    tick_scale           = stats['tick_scale']
    tick_track_raw       = stats['tick_track_raw']         # pre-EMA
    tick_effort_raw      = stats['tick_effort_raw']        # pre-criticality, pre-EMA
    tick_effort_weighted = stats['tick_effort_weighted']   # post-criticality, pre-EMA
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

    # extract signals for the chosen (drone, seed). EMA'd are what training saw.
    # for effort: show WEIGHTED (post-criticality, pre-EMA) as the "raw"-equivalent
    # line, since the criticality gate is part of the signal pipeline. raw raw
    # (pre-criticality) is no longer plotted — too misleading once criticality is on.
    fit_signal             = tick_fit            [best_d, best_s]      # dt-scaled reward increments
    scale_signal           = tick_scale          [best_d, best_s] / dt
    track_signal           = tick_track          [best_d, best_s] / dt # ema'd
    effort_signal          = tick_effort         [best_d, best_s] / dt # ema'd
    track_raw_signal       = tick_track_raw      [best_d, best_s] / dt # raw
    effort_weighted_signal = tick_effort_weighted[best_d, best_s] / dt # post-criticality, pre-EMA

    cum_fit = np.cumsum(fit_signal)
    time    = np.arange(T) * dt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(time, cum_fit, color='C0')
    ax.set_title(f'cumulative fitness — drone {best_d}, seed {best_s} (final={cum_fit[-1]:.3f})')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('cum fitness')
    ax.grid(alpha=0.3)

    def plot_ema_with_raw(ax, raw, ema, color, title, ylabel, raw_label='raw'):
        ax.plot(time, raw, color='gray', alpha=0.3, linewidth=0.6, label=raw_label)
        ax.plot(time, ema, color=color, linewidth=1.3,            label='ema (training)')
        ax.set_title(title)
        ax.set_xlabel('time (s)')
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)

    plot_ema_with_raw(axes[0, 1], track_raw_signal,       track_signal,  'C1',
                      'track quality (training signal)',  'track',  raw_label='raw')
    plot_ema_with_raw(axes[1, 0], effort_weighted_signal, effort_signal, 'C2',
                      'effort quality (weighted vs weighted+EMA)', 'effort', raw_label='weighted')

    # scale has no EMA — single line
    ax = axes[1, 1]
    ax.plot(time, scale_signal, color='C3', linewidth=1.0)
    ax.set_title('scale')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('scale')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

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
        bins = np.linspace(0, 1, 41)
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

        ax[1, 2].axis('off')   # autocorr removed — leave slot empty

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
    # diagonal:        per-signal histogram
    # lower triangle:  scatter of (col-signal x, row-signal y)
    # upper triangle:  noise of row-signal binned by col-signal value
    #                  -> answers "is signal_row noisier when signal_col is high/low?"
    # column names labeled along the top row, row names along the left column.
    # =========================================================================
    fit_per_tick = fit_signal / dt   # per-tick reward (track * effort * scale * pretouch)

    pair_signals = [
        ('track',    best_trk,     'C1'),
        ('effort',   best_eff,     'C2'),
        ('scale',    best_scl,     'C3'),
        ('fit/tick', fit_per_tick, 'C0'),
    ]
    n_sig = len(pair_signals)

    # windowed means for the lower-triangle scatter — matches signal_diagnostic's
    # W=32-tick scatter so the same plot in both figures shows the same data.
    W_pair = 32
    def windowed_mean(x: np.ndarray, W: int = W_pair) -> np.ndarray:
        n = x.size - W
        return np.array([x[t:t + W].mean() for t in range(n)], dtype=np.float32) if n > 0 else x

    pair_windowed = [(name, windowed_mean(sig), color) for name, sig, color in pair_signals]

    # per-(drone, seed) episode means for the upper-triangle scatter — one point per
    # (drone, seed). reveals whether (sig_i, sig_j) plane *separates* good drones from
    # bad ones, which is the actual selection question. complements the lower triangle
    # (which is single-drone within-episode behavior).
    # per-(drone, seed) episode means of each signal.
    ep_arrays = {
        'track':    (tick_track  / dt).mean(axis=2).flatten(),
        'effort':   (tick_effort / dt).mean(axis=2).flatten(),
        'scale':    (tick_scale  / dt).mean(axis=2).flatten(),
        'fit/tick': (tick_fit    / dt).mean(axis=2).flatten(),
    }
    ep_drone_idx = np.broadcast_to(np.arange(N)[:, None], (N, S)).flatten()
    ep_drone_fit = drone_se_fit[ep_drone_idx]    # color = SE fitness of that drone

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
    for i, (name_i, sig_i, color_i) in enumerate(pair_signals):
        for j, (name_j, sig_j, color_j) in enumerate(pair_signals):
            a = axes3[i, j]
            if i == j:
                # DIAGONAL: histogram
                a.hist(sig_i, bins=40, color=color_i, alpha=0.7)
                a.set_xlim(-0.05, 1.05)
                a.set_yticks([])
            elif i > j:
                # LOWER TRIANGLE: scatter of windowed means (matches signal_diagnostic).
                _, win_i, _ = pair_windowed[i]
                _, win_j, _ = pair_windowed[j]
                a.scatter(win_j, win_i, color='gray', alpha=0.4, s=4)
                r = float(np.corrcoef(win_j, win_i)[0, 1]) if win_i.size > 1 else float('nan')
                a.text(0.03, 0.97, f'r={r:+.2f} (W={W_pair})', transform=a.transAxes,
                       va='top', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                a.set_xlim(-0.05, 1.05)
                a.set_ylim(-0.05, 1.05)
            else:
                # UPPER TRIANGLE: per-(drone, seed) episode means, colored by drone fitness.
                ep_x = ep_arrays[name_j]
                ep_y = ep_arrays[name_i]
                a.scatter(ep_x, ep_y, c=ep_drone_fit, cmap='viridis',
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
                a.set_xlim(-0.05, 1.05)
                a.set_ylim(-0.05, 1.05)

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
                  f'upper = per-(drone, seed) episode means (across drones, color = SE fitness)',
                  fontsize=10)
    fig3.tight_layout()
    fig3.savefig('diagnostic_pairs.png', dpi=150)

    plt.show()
