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
        seed = (int(np.random.randint(0, 100)), int(np.random.randint(0, 100)))

    train_seed, eval_seed = seed

    # backfill keys for older checkpoints
    settings.setdefault('perturbations', False)
    settings.setdefault('curriculum_progress', 0)

    simulator = parallel_sim
    # simulator = sim

    with open('config.toml', 'rb') as f:
        config = tomllib.load(f)
    population = config['trainer']['population']

    progress_old = None
    # load checkpoint
    # create Mp Pool
    with Pool(initializer=_worker_init) as Mpool:
        # generate simulation seed pool
        try:
            run = True
            print('Training Start')
            while run:
                t0 = time.perf_counter()

                # rotate training seed every 10 gens (eval_seed stays fixed until curriculum advance)
                if alg.gen > 0 and alg.gen % 10 == 0:
                    train_seed = int(np.random.randint(0, 100))

                # propose networks          -> returns drone individuals + stats
                tp0 = time.perf_counter()
                individuals, propose_stats = alg.propose(population, None)
                t_prop = time.perf_counter() - tp0

                # score individuals         -> updates individuals + returns stats
                ts0 = time.perf_counter()
                scores, sim_stats = simulator(individuals, settings, Mpool, seed=train_seed)
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
                print(f"  curriculum: length {settings['length']:>5.2f} | limit {settings['limit']:>5.2f} | progress {settings['curriculum_progress']} | perturbations {settings['perturbations']} | train_seed {train_seed} | eval_seed {eval_seed}", flush=True)

                # pool transition branch:
                if alg.gen % 50 == 0:
                    # measure progress on the held-out eval seed using current top-10 archive elites.
                    # decouples gating from train-seed rotation noise. eval results are discarded —
                    # archive is not mutated by this pass (workers operate on pickled copies).
                    fit_flat = alg.archive.fit.ravel()
                    indv_flat = alg.archive.indv.ravel()
                    finite_mask = np.isfinite(fit_flat)
                    if finite_mask.sum() >= 1:
                        top_idx = np.argsort(fit_flat[finite_mask])[-10:]
                        top_elites = [indv_flat[finite_mask.nonzero()[0][i]] for i in top_idx]
                        _, eval_stats = simulator(top_elites, settings, Mpool, seed=eval_seed)
                        progress = eval_stats['fit_mean']
                    else:
                        progress = 0.0

                    if progress_old is None:
                        progress_old = progress

                    print(f"  gate:       eval_progress {progress:.3f} (prev {progress_old:.3f})", flush=True)

                    # absolute-delta plateau gate. 0.01 in log-fitness space ≈ 1% improvement
                    # in pre-log space (log(1.01) ≈ 0.01). robust to small progress_old values.
                    if progress_old > 0 and (progress - progress_old) / progress_old < 0.01:
                        # update curriculum at pool end
                        if settings['length'] < MAX_LENGTH:
                            settings['length'] *= 1.05
                        # count progression; enable perturbations after the 2nd advance
                        settings['curriculum_progress'] += 1
                        if settings['curriculum_progress'] >= 2:
                            settings['perturbations'] = True
                        # regenerate both seeds: train rotates, eval gets a fresh measuring stick for the new length
                        train_seed = int(np.random.randint(0, 100))
                        eval_seed  = int(np.random.randint(0, 100))
                        # revalidate existing best drones at pool end
                        elites = alg.archive.pop()
                        elites, _ = simulator(elites, settings, Mpool, seed=train_seed)
                        alg.reset(elites)

                        # reset progress baseline so the next gate measures against the new stage
                        progress_old = None

                        print(f"  -> curriculum transition: length={settings['length']:.2f}  progress={settings['curriculum_progress']}  perturbations={settings['perturbations']}  elites={len(elites)}  new train_seed={train_seed} eval_seed={eval_seed}", flush=True)
                    else:
                        progress_old = progress

                # save checkpoint at generation threshold/new best
                if alg.gen % 25 == 0:
                    save(save_path, alg, settings, (train_seed, eval_seed))
                    print(f"  -> checkpoint saved (gen {alg.gen})", flush=True)

        except KeyboardInterrupt:
            print('Terminating Training')
            save(save_path, alg, settings, (train_seed, eval_seed))
            print('Saved at', save_path)
            show_archive(alg)
