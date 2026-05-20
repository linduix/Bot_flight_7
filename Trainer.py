from modules.individual import Individual
from modules.evo_alg.mapElites import algorithm, save, load
from modules.simulation.sim1 import sim
import numpy as np
import tomllib
import os


def show_archive(alg):
    import matplotlib.pyplot as plt
    occ_mask = alg.archive.fit > -np.inf
    extent   = [alg.archive.xrange[0], alg.archive.xrange[1],
                alg.archive.yrange[0], alg.archive.yrange[1]]

    panels = [
        ('fitness',       alg.archive.fit,  'viridis'),
        ('log curiosity', alg.archive.curi, 'magma'),
        ('improvement',   alg.archive.impr, 'plasma'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    for ax, (label, data, cmap) in zip(axes, panels):
        display = np.where(occ_mask, data, np.nan)
        im = ax.imshow(display.T, origin='lower', aspect='auto', cmap=cmap, extent=extent) # type: ignore
        ax.set_box_aspect(1)
        plt.colorbar(im, ax=ax, label=label)
        ax.set_xlabel('mean |angular velocity| (rad/s)')
        ax.set_ylabel('mean thrust saturation')
        ax.set_title(f'{label} — gen {alg.gen}')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig('archive_heatmap.png', dpi=150)
    plt.show()

save_path = os.path.join('data', 'MAP_Checkpoint.pkl')

if __name__=='__main__':
    # load save
    if os.path.isfile(save_path):
        alg, settings, seeds = load(save_path)
    else:
        alg = algorithm(resolution=50)
        settings  = {'limit': 10.0, 'length': 10.0}
        seeds = np.random.randint(0, 100, size=5)

    simulator = sim

    with open('config.toml', 'rb') as f:
        config = tomllib.load(f)
    population = config['trainer']['population']

    # load checkpoint
    # create Mp Pool
    # generate simulation seed pool
    try:
        run = True
        print('Training Start')
        while run:
            # propose networks          -> returns drone individuals + stats
            individuals, propose_stats = alg.propose(population, None)

            # score individuals         -> updates individuals + returns stats
            seed = np.random.choice(seeds)
            sim_stats = simulator(individuals, settings, seed=seed)

            # send results to algorithm -> returns stats
            update_stats = alg.update(individuals)

            # emit stats to logging
            best = update_stats['archive_best'] if update_stats['archive_best'] is not None else 0.0
            cov  = update_stats['coverage']
            upd  = update_stats['updates']

            print(f"\n=== gen {alg.gen} ===")

            # propose: per-arm budget allocation + bandit performance relative to best arm
            budget = propose_stats['budget']
            budget_str = ' | '.join(f"{k} {v:>4d}" for k, v in budget.items())
            print(f"  propose:    {budget_str}")
            means = {arm: (s['score']/s['pulls'] if s['pulls'] > 0 else 0.0) for arm, s in alg.bandit.arms.items()}
            top_mean = max(means.values()) if means else 0.0
            print(f"              {'arm':<10} {'mean':>7} {'ratio':>7}")
            for arm, m in means.items():
                ratio = m/top_mean if top_mean > 0 else 0.0
                print(f"              {arm:<10} {m:>7.3f} {ratio:>6.2f}x")

            # sim: per-batch fitness
            print(f"  sim:        fit_mean {sim_stats['fit_mean']:>6.3f} | fit_max {sim_stats['fit_max']:>6.3f}")

            # update: archive churn + bandit reward this gen
            score_str = ' | '.join(f"{k} {int(v):>3d}" for k, v in update_stats['bandit_score'].items())
            print(f"  update:     coverage {cov:>4.2f} | updates {upd:>4d} | best {best:>6.3f}")
            print(f"              bandit_score: {score_str}", flush=True)

            # curriculum: current difficulty
            print(f"  curriculum: length {settings['length']:>5.2f} | limit {settings['limit']:>5.2f}", flush=True)

            # pool transition branch:
            if update_stats['archive_best'] / settings['limit'] > 0.8:
                # update curriculum at pool end
                settings['length'] *= 1.05
                # regenerate seed pool
                seeds = np.random.randint(0, 100, size=5)
                # revalidate existing best drones at pool end
                elites = alg.archive.pop()
                simulator(elites, settings, seed=seeds[0])
                alg.reset(elites)
                print(f"  -> curriculum transition: length={settings['length']:.2f}  elites={len(elites)}", flush=True)

            # save checkpoint at generation threshold/new best
            if alg.gen % 50 == 0:
                save(save_path, alg, settings, seeds)
                print(f"  -> checkpoint saved (gen {alg.gen})", flush=True)

    except KeyboardInterrupt:
        print('Terminating Training')
        save(save_path, alg, settings, seeds)
        print('Saved at', save_path)
        show_archive(alg)
