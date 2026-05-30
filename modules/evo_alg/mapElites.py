from modules.individual import Individual
from modules.evo_alg import arms
from typing import Callable, cast
import pickle as pkl
import numpy as np
import os

class MAB():
    def __init__(self, arms: list[str]) -> None:
        self.decay = 0.95
        self.total_pulls = 0
        self.arms = {}
        for k in arms:
            self.arms[k] = {'pulls': 0.0, 'score': 0.0, 'value': np.inf}

    def update_stats(self, scores: dict[str, float]):
        # update the new scores
        for arm, score in scores.items():
            self.arms[arm]['score'] += score

        # decay the old values
        for stats in self.arms.values():
            stats['pulls'] *= self.decay
            stats['score'] *= self.decay
        self.total_pulls *= self.decay

    def recompute_values(self):
        # calculate arm value
        for _, stats in self.arms.items():
            if stats['pulls'] == 0:
                continue

            # arm_value = mean score + sqrt( 2 * ln(total pulls) / arm pulls )
            mean_score  = stats['score'] / stats['pulls']
            exploration = np.sqrt(0.02 * np.log(self.total_pulls) / stats['pulls'])
            arm_value   = mean_score + exploration

            stats['value'] = arm_value

    def pull(self, qty: int) -> dict[str, int]:
        budget = {k: 0 for k in self.arms}

        for _ in range(qty):
            self.recompute_values()
            # get for arm with most value
            best = max(self.arms, key=lambda k: self.arms[k]['value'])

            # increment the budget + pulls
            budget[best]     += 1
            self.total_pulls += 1
            self.arms[best]['pulls'] += 1

        return budget

class Archive():
    def __init__(self, res) -> None:
        # descriptor minmax; x: mean gimble angle, y: activation variance
        self.xrange: tuple = (0.20, 0.75)
        self.yrange: tuple = (0.02, 0.25)

        # archive matrices
        self.res  = res
        self.indv:    np.ndarray = np.empty((res, res), dtype=object) # matrix of Individuals
        self.fit :    np.ndarray = np.full((res, res),  -np.inf)      # fitness matrix
        self.curi:    np.ndarray = np.full((res, res),  0.01000)      # curiosity matrix
        self.impr:    np.ndarray = np.full((res, res),  0.01000)      # fitness improvement matrix
        self.updates: np.ndarray = np.full((res, res),  0)             # updates tracking matrix

        self.curi_decay = 0.99

    def coordinates(self, i: Individual) -> tuple[int, int]:
        # calculate the archive coordinates for individual
        xval, yval = i.descriptors['mean_gimb'], i.descriptors['var_action']
        idx = int(((xval - self.xrange[0]) / (self.xrange[1] - self.xrange[0])) * self.res)
        idx = np.clip(idx, 0, self.res - 1) # keep index in bounds

        idy = int(((yval - self.yrange[0]) / (self.yrange[1] - self.yrange[0])) * self.res)
        idy = np.clip(idy, 0, self.res - 1) # keep index in bounds

        return idx, idy

    def insert(self, i: Individual) -> float:
        idx, idy = self.coordinates(i)
        old_fit = self.fit[idx, idy]

        if i.fitness <= old_fit:
            if i.parent_idx is not None:
                self.curi[i.parent_idx] = max(self.curi[i.parent_idx] * self.curi_decay, 0.01)
            return 0.0

        # update the archivess
        self.fit[idx, idy]  =  i.fitness
        self.indv[idx, idy] =  i
        if i.parent_idx is not None:
            self.curi[i.parent_idx] += 1
        if old_fit != -np.inf:
            self.impr[idx, idy] += i.fitness - old_fit

        return i.fitness - old_fit if old_fit != -np.inf else i.fitness

    def get(self, row: int, col: int) -> Individual:
        return cast(Individual, self.indv[row, col])

    def pop(self) -> list[Individual]:
        indv = [x for x in self.indv.flat if x is not None]
        return cast(list[Individual], indv)

class algorithm():
    def __init__(self, resolution) -> None:
        self.gen = 0
        self.archive = Archive(resolution)
        self.cma = arms.cma()

        self.arms: dict['str', Callable[[Archive, int], list[Individual]]] = {
            'random'  : arms.random,
            'gaussian': arms.gaussian,
            'iso'     : arms.iso,
            'cma'     : self.cma.ask
        }
        self.bandit = MAB(list(self.arms.keys()))

    def propose(self, qty, Mpool=None) -> tuple[ list[Individual], dict ]:
        # initial bootsrap
        self.batch_size = qty
        if self.gen == 0:
            proposition: list[Individual] = arms.random(self.archive, qty)
            return proposition, {'budget': {'random': qty}}

        # get the arm budget
        budget = self.bandit.pull(qty)
        proposition: list[Individual] = []

        # for each arm add its budgeted propositions
        for arm, budg in budget.items():
            indvs = self.arms[arm](self.archive, budg)
            proposition.extend(indvs)

        return proposition, {'budget': dict(budget)}


    def update(self, individuals: list[Individual]):
        bandit_score = {k: 0.0 for k in self.arms.keys()}
        self.archive.curi_decay = 0.1 ** (8/self.batch_size) # curiosity decay factor

        stats = {'discoveries': 0, 'updates': 0, 'bandit_score': bandit_score}
        for i in individuals:
            idx, idy = self.archive.coordinates(i)
            was_empty = self.archive.fit[idx, idy] == -np.inf
            delta = self.archive.insert(i)

            # if successful update, reward bandit:
            if delta > 0:
                if was_empty:
                    stats['discoveries'] += 1
                else:
                    stats['updates'] += 1

                if i.tag in bandit_score:
                    old_fit = i.fitness - delta
                    bandit_score[i.tag] += i.fitness**2 - max(old_fit, 0.0)**2

        # update cma
        self.cma.tell(individuals)

        # update the bandit
        if self.gen > 0:
            self.bandit.update_stats(bandit_score)

        # decay improvement matrix
        self.archive.impr *= 0.95
        self.archive.impr = np.maximum(0.01, self.archive.impr)

        # smoke-test diagnostics: is the archive actually filling?
        occupied = self.archive.fit > -np.inf
        stats['coverage']     = int(occupied.sum()) / (self.archive.res * self.archive.res)
        stats['archive_best'] = float(self.archive.fit[occupied].max()) if occupied.any() else None

        # increment gen
        self.gen += 1
        return stats

    def reset(self, initial_pop: list[Individual]):
        resoulution  = self.archive.res
        self.archive = Archive(resoulution)

        self.bandit      = MAB(list(self.arms.keys()))
        self.cma         = arms.cma()
        self.arms['cma'] = self.cma.ask

        for i in initial_pop:
            i.parent_idx = None
            self.archive.insert(i)

def save(path, alg, settings, seed):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pkl.dump({'alg': alg, 'settings': settings, 'seed': seed}, f)
    os.replace(tmp, path)

def load(path) -> tuple:
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data['alg'], data['settings'], data['seed']

import signal as _signal
def _worker_init():
    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)

if __name__ == "__main__":
    import time
    from multiprocessing.pool import Pool
    from modules.simulation import sim1

    N_SEEDS     = 1
    MAX_SECONDS = 60 * 5
    BATCH_SIZE  = 4000
    RESOLUTION  = 25

    results = []
    with Pool(initializer=_worker_init) as Mpool:
        for k in range(N_SEEDS):
            print(f"\n=== seed {k+1}/{N_SEEDS} ===", flush=True)
            alg = algorithm(RESOLUTION)

            start = time.time()
            gen_count = 0
            trajectory = []
            total_evals = 0
            while time.time() - start < MAX_SECONDS:
                indv, propstat = alg.propose(BATCH_SIZE)
                indv, simstat  = sim1.parallel_sim(indv, {'limit': 10, 'length': 10}, Mpool)
                updatestat     = alg.update(indv)
                elapsed = time.time() - start
                total_evals += BATCH_SIZE

                occ = alg.archive.fit[alg.archive.fit > -np.inf]
                top10 = np.sort(occ)[-10:].mean() if len(occ) >= 10 else occ.mean()
                trajectory.append((elapsed, updatestat['archive_best'], top10))
                gen_count += 1

            times  = np.array([t[0] for t in trajectory])
            bests  = np.array([t[1] for t in trajectory])
            top10s = np.array([t[2] for t in trajectory])

            auc_best  = float(np.trapezoid(bests,  times) / times[-1])
            auc_top10 = float(np.trapezoid(top10s, times) / times[-1])

            time_to = {}
            for thr in [2.0, 3.0, 4.0, 5.0]:
                idx = np.argmax(bests >= thr)
                time_to[thr] = float(times[idx]) if bests[idx] >= thr else None

            total_pulls = sum(s['pulls'] for s in alg.bandit.arms.values())
            total_score = sum(s['score'] for s in alg.bandit.arms.values())
            arm_stats = {}
            for arm, s in alg.bandit.arms.items():
                arm_stats[arm] = {
                    'mean':  s['score']/s['pulls'] if s['pulls'] > 0 else 0.0,
                    'pull%': s['pulls']/total_pulls if total_pulls > 0 else 0.0,
                    'score%':s['score']/total_score if total_score > 0 else 0.0,
                }

            coverage = float((alg.archive.fit > -np.inf).sum()) / (alg.archive.res ** 2)

            results.append({
                'gens':        gen_count,
                'final_top10': float(top10s[-1]),
                'auc_best':    auc_best,
                'auc_top10':   auc_top10,
                'coverage':    coverage,
                'time_to':     time_to,
                'arms':        arm_stats,
                'alg':         alg,
            })
            print(f"  top10={results[-1]['final_top10']:.3f}  AUC_top10={results[-1]['auc_top10']:.3f}  cov={coverage:.2f}  gens={gen_count}", flush=True)

    # ---- summary stats across seeds ----
    top10s = [r['final_top10'] for r in results]
    aucs   = [r['auc_top10']   for r in results]
    aucbs  = [r['auc_best']    for r in results]
    covs   = [r['coverage']    for r in results]
    gens   = [r['gens']        for r in results]

    print(f"\n=== summary across {N_SEEDS} seeds ({MAX_SECONDS}s, batch={BATCH_SIZE}) ===")
    print(f"  {'metric':<14} {'min':>7} {'median':>7} {'max':>7} {'mean':>7} {'std':>7}")
    for name, vals in [('final_top10', top10s), ('AUC_top10', aucs), ('AUC_best', aucbs), ('coverage', covs), ('gens', gens)]:
        a = np.array(vals, dtype=float)
        print(f"  {name:<14} {a.min():>7.3f} {np.median(a):>7.3f} {a.max():>7.3f} {a.mean():>7.3f} {a.std():>7.3f}")

    print(f"\n  threshold reach rate:")
    for thr in [2.0, 3.0, 4.0, 5.0]:
        hit_times = [r['time_to'][thr] for r in results if r['time_to'][thr] is not None]
        rate = len(hit_times) / N_SEEDS
        if hit_times:
            print(f"    fit>={thr}: {rate:.0%}  (median time {np.median(hit_times):.1f}s)")
        else:
            print(f"    fit>={thr}: {rate:.0%}  (never reached)")

    print(f"\n  arm allocation (mean across seeds):")
    print(f"    {'arm':<10} {'pull%':>7} {'score%':>7} {'mean':>7}")
    for arm in results[0]['arms']:
        pull_pct  = np.mean([r['arms'][arm]['pull%']  for r in results])
        score_pct = np.mean([r['arms'][arm]['score%'] for r in results])
        mean_sc   = np.mean([r['arms'][arm]['mean']   for r in results])
        print(f"    {arm:<10} {pull_pct:>6.1%} {score_pct:>6.1%} {mean_sc:>7.3f}")

    # ---- heatmap from the final seed run ----
    alg = results[-1]['alg']

    import matplotlib.pyplot as plt
    occ_mask = alg.archive.fit > -np.inf
    extent   = [alg.archive.xrange[0], alg.archive.xrange[1],
                alg.archive.yrange[0], alg.archive.yrange[1]]

    panels = [
        ('fitness',       alg.archive.fit,          'viridis'),
        ('log curiosity', alg.archive.curi,         'magma'),
        ('improvement',   alg.archive.impr,         'plasma'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    for ax, (label, data, cmap) in zip(axes, panels):
        display = np.where(occ_mask, data, np.nan)
        im = ax.imshow(display.T, origin='lower', aspect='auto', cmap=cmap, extent=extent) # type: ignore
        ax.set_box_aspect(1)
        plt.colorbar(im, ax=ax, label=label)
        ax.set_xlabel('mean gimbal angle (rad)')
        ax.set_ylabel('activation variance')
        ax.set_title(f'{label} — gen {alg.gen} (final seed)')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig('archive_heatmap.png', dpi=150)
    plt.show()
