from modules.individual import Individual
from modules.evo_alg.mapElites import algorithm, save, load
from modules.simulation.sim1 import sim, parallel_sim
from multiprocessing.pool import Pool
import numpy as np
import tomllib
import os
import time
import signal


def _worker_init():
    # ignore SIGINT in workers — only main process handles Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def show_archive(alg):
    import matplotlib.pyplot as plt
    occ_mask = alg.archive.fit > -np.inf
    extent   = [alg.archive.xrange[0], alg.archive.xrange[1],
                alg.archive.yrange[0], alg.archive.yrange[1]]

    fail_success = np.log10((alg.archive.failed + 1) / (alg.archive.successes + 1))

    panels = [
        ('fitness',       alg.archive.fit,  'viridis'),
        ('log curiosity', alg.archive.curi, 'magma'),
        ('improvement',   alg.archive.impr, 'plasma'),
        ('fail/success',  fail_success,     'inferno'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    axes = axes.flatten()
    for ax, (label, data, cmap) in zip(axes, panels):
        display = np.where(occ_mask, data, np.nan)
        im = ax.imshow(display.T, origin='lower', aspect='auto', cmap=cmap, extent=extent) # type: ignore
        ax.set_box_aspect(1)
        plt.colorbar(im, ax=ax, label=label)
        ax.set_xlabel('mean gimbal angle (rad)')
        ax.set_ylabel('activation variance')
        ax.set_title(f'{label} — gen {alg.gen}')

    plt.subplots_adjust(left=0.08, right=0.96, top=0.93, bottom=0.08, hspace=0.45, wspace=0.3)
    plt.savefig('archive_heatmap.png', dpi=150)
    plt.show()

save_path = os.path.join('data', 'MAP_Checkpoint.pkl')

RESOLUTION = 10
MAX_LENGTH = 75

if __name__=='__main__':
    # load save
    if os.path.isfile(save_path):
        archive, gen, settings, seed = load(save_path)
        alg = algorithm(resolution=RESOLUTION)
        alg.archive = archive
        alg.archive.res = RESOLUTION
        alg.gen = gen

    else:
        alg = algorithm(resolution=RESOLUTION)
        settings  = {'limit': 10.0, 'length': 10.0, 'perturbations': False, 'curriculum_progress': 0}
        seed = np.random.randint(0, 100)

    # backfill keys for older checkpoints
    settings.setdefault('perturbations', False)
    settings.setdefault('curriculum_progress', 0)

    simulator = parallel_sim
    # simulator = sim

    with open('config.toml', 'rb') as f:
        config = tomllib.load(f)
    population = config['trainer']['population']

    top10_old = None
    # load checkpoint
    # create Mp Pool
    with Pool(initializer=_worker_init) as Mpool:
        # generate simulation seed pool
        try:
            run = True
            print('Training Start')
            while run:
                t0 = time.perf_counter()

                # propose networks          -> returns drone individuals + stats
                tp0 = time.perf_counter()
                individuals, propose_stats = alg.propose(population, None)
                t_prop = time.perf_counter() - tp0

                # score individuals         -> updates individuals + returns stats
                ts0 = time.perf_counter()
                scores, sim_stats = simulator(individuals, settings, Mpool, seed=seed)
                # scores, sim_stats = simulator(individuals, settings, seed=seed)
                t_sim = time.perf_counter() - ts0

                # send results to algorithm -> returns stats
                tu0 = time.perf_counter()
                update_stats = alg.update(scores)
                t_upd = time.perf_counter() - tu0

                elapsed = time.perf_counter() - t0

                # emit stats to logging
                best = update_stats['archive_best'] if update_stats['archive_best'] is not None else 0.0
                cov  = update_stats['coverage']
                disc = update_stats['discoveries']
                upd  = update_stats['updates']

                print(f"\n=== gen {alg.gen} [{elapsed:.2f}s] ===")
                print(f"  timing:     prop {t_prop:>5.2f}s | sim {t_sim:>5.2f}s | upd {t_upd:>5.2f}s")

                # propose: active emitter counts
                ps = propose_stats
                active_str = f"active {ps['active']}  " + '  '.join(f"{k} {v}" for k, v in ps.items() if k != 'active')
                print(f"  propose:    {active_str}")

                # sim: per-batch fitness
                print(f"  sim:        fit_mean {sim_stats['fit_mean']:>6.3f} | fit_max {sim_stats['fit_max']:>6.3f}")

                # update: archive churn + bandit reward this gen
                score_str = ' | '.join(f"{k} {v:>6.2f}" for k, v in update_stats['bandit_score'].items())
                print(f"  update:     coverage {cov:>4.2f} | discoveries {disc:>4d} | updates {upd:>4d} | best {best:>6.3f}")
                print(f"              improvement: {score_str}", flush=True)

                # curriculum: current difficulty
                finite = alg.archive.fit[np.isfinite(alg.archive.fit)]
                top10 = np.sort(finite)[-10:].mean() if finite.size >= 10 else (finite.mean() if finite.size else 0.0)
                if top10_old is None:
                    top10_old = top10
                print(f"  curriculum: length {settings['length']:>5.2f} | limit {settings['limit']:>5.2f} | progress {settings['curriculum_progress']} | perturbations {settings['perturbations']} | top10 {top10:.2f}", flush=True)

                # pool transition branch:
                if alg.gen % 50 == 0:
                    # absolute-delta plateau gate. 0.01 in log-fitness space ≈ 1% improvement
                    # in pre-log space (log(1.01) ≈ 0.01). robust to small top10_old values.
                    # raise threshold if gen-to-gen noise on top10 turns out > 0.01.
                    if (top10 - top10_old) < 0.01 and top10 > 0:
                        # update curriculum at pool end
                        if settings['length'] < MAX_LENGTH:
                            settings['length'] *= 1.05
                        # count progression; enable perturbations after the 2nd advance
                        settings['curriculum_progress'] += 1
                        if settings['curriculum_progress'] >= 2:
                            settings['perturbations'] = True
                        # regenerate seed pool
                        seed = np.random.randint(0, 100)
                        # revalidate existing best drones at pool end
                        elites = alg.archive.pop()
                        elites, _ = simulator(elites, settings, Mpool, seed=seed)
                        alg.reset(elites)

                        finite = alg.archive.fit[np.isfinite(alg.archive.fit)]
                        top10 = np.sort(finite)[-10:].mean() if finite.size >= 10 else (finite.mean() if finite.size else 0.0)

                        print(f"  -> curriculum transition: length={settings['length']:.2f}  progress={settings['curriculum_progress']}  perturbations={settings['perturbations']}  elites={len(elites)}", flush=True)
                    top10_old = top10

                # save checkpoint at generation threshold/new best
                if alg.gen % 25 == 0:
                    save(save_path, alg, settings, seed)
                    print(f"  -> checkpoint saved (gen {alg.gen})", flush=True)

        except KeyboardInterrupt:
            print('Terminating Training')
            save(save_path, alg, settings, seed)
            print('Saved at', save_path)
            show_archive(alg)
