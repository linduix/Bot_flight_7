from modules.evo_alg.mapElites import load
from modules.simulation.sim1 import sim
import numpy as np
import matplotlib.pyplot as plt
import os


save_path = os.path.join('data', 'MAP_Checkpoint.pkl')


def pick_3x3_sector_elites(archive):
    # Mirror sim1_visual.py's 3x3 sector selection: split the archive grid into
    # 9 blocks, pick the highest-fitness occupant per block. Skips empty blocks.
    grid = archive.indv
    R, C = grid.shape
    K = 3
    elites = []
    for i in range(K):
        for j in range(K):
            block = grid[i*R//K:(i+1)*R//K, j*C//K:(j+1)*C//K]
            cands = [x for x in block.flat if x is not None]
            if cands:
                elites.append(max(cands, key=lambda x: x.fitness))
    return elites


if __name__ == "__main__":
    alg, settings, seed = load(save_path)
    print(f"loaded checkpoint (gen {alg.gen}) — settings={settings} seed={seed}")

    elites = pick_3x3_sector_elites(alg.archive)
    print(f"picked {len(elites)} elites via 3x3 sector selection")

    # run the sim with per-tick logging on the saved seed → same regime as training
    _, stats = sim(elites, settings, seed=seed, log_per_tick=True)

    tick_fit    = stats['tick_fit']     # (N, S, T)
    tick_track  = stats['tick_track']
    tick_effort = stats['tick_effort']
    tick_scale  = stats['tick_scale']
    dt          = stats['dt']
    N, S, T     = tick_fit.shape

    # best drone = highest seed-mean fitness; best seed within that drone = highest episode fitness
    drone_fit    = tick_fit.sum(axis=2)              # (N, S) episode totals
    drone_mean   = drone_fit.mean(axis=1)            # (N,)
    best_d       = int(drone_mean.argmax())
    best_s       = int(drone_fit[best_d].argmax())
    best_fitness = float(drone_fit[best_d, best_s])

    ind = elites[best_d]
    print(f"best drone idx={best_d} (tag={ind.tag}) seed={best_s} "
          f"fitness={best_fitness:.3f} mean_seed_fit={drone_mean[best_d]:.3f} "
          f"descr={ind.descriptors}")

    # extract the four signals for the chosen (drone, seed)
    fit_signal    = tick_fit   [best_d, best_s]      # dt-scaled increments
    track_signal  = tick_track [best_d, best_s] / dt # raw quality ∈ [0, 1]
    effort_signal = tick_effort[best_d, best_s] / dt
    scale_signal  = tick_scale [best_d, best_s] / dt

    cum_fit = np.cumsum(fit_signal)
    ticks   = np.arange(T)
    time    = ticks * dt

    # chatter density — rolling mean of |Δx|. linear weighting so a single big
    # jump contributes mag/W (~0.03 for W=32) instead of dominating the window.
    # constant low-amplitude noise correctly reads near its amplitude.
    W = 32
    kernel = np.ones(W, dtype=np.float32) / W
    def chatter_density(x: np.ndarray) -> np.ndarray:
        d = np.abs(np.diff(x, prepend=x[0]))
        return np.convolve(d, kernel, mode='same')

    track_noise  = chatter_density(track_signal)
    effort_noise = chatter_density(effort_signal)
    scale_noise  = chatter_density(scale_signal)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(time, cum_fit, color='C0')
    ax.set_title(f'cumulative fitness — drone {best_d}, seed {best_s} (final={cum_fit[-1]:.3f})')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('cum fitness')
    ax.grid(alpha=0.3)

    def plot_with_noise(ax, signal, noise, color, title, ylabel):
        ax.plot(time, signal, color=color, linewidth=0.8, label=ylabel)
        ax.set_title(title)
        ax.set_xlabel('time (s)')
        ax.set_ylabel(ylabel, color=color)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(time, noise, color='gray', alpha=0.5, linewidth=1.0)
        ax2.set_ylabel('chatter (mean |Δ|)', color='gray')
        return ax2

    ax_track  = plot_with_noise(axes[0, 1], track_signal,  track_noise,  'C1',
                                'track quality / tick (∈ [0, 1])',  'track')
    ax_effort = plot_with_noise(axes[1, 0], effort_signal, effort_noise, 'C2',
                                'effort quality / tick (∈ [0, 1])', 'effort')
    ax_scale  = plot_with_noise(axes[1, 1], scale_signal,  scale_noise,  'C3',
                                'scale (prox * pretouch) / tick',   'scale')

    # shared chatter y-axis: auto-scale to the global max across all three signals
    chatter_max = max(track_noise.max(), effort_noise.max(), scale_noise.max())
    chatter_max = float(chatter_max) * 1.05
    for a in (ax_track, ax_effort, ax_scale):
        a.set_ylim(0, chatter_max)

    plt.suptitle(f'diagnostic — best of 3x3 elites (gen {alg.gen}, seed {seed})')
    plt.tight_layout()
    plt.savefig('diagnostic.png', dpi=150)
    plt.show()
