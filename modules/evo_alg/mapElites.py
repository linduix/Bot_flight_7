from modules.individual import Individual
from modules.evo_alg import arms
from typing import Callable
import numpy as np

class MAB():
    def __init__(self, arms: list[str]) -> None:
        self.decay = 0.95
        self.total_pulls = 0
        self.arms = {}
        for k in arms:
            self.arms[k] = {'pulls': 0, 'score': 0, 'value': np.inf}

    def update_stats(self, scores: dict[str, float]):
        # update the new scores
        for arm, score in scores.items():
            self.arms[arm]['score'] += score

        # decay the old values
        for _, stats in self.arms.items():
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
            exploration = np.sqrt(2 * np.log(self.total_pulls) / stats['pulls'])
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


class algorithm():
    def __init__(self, resolution) -> None:
        self.gen = 0
        self.res = resolution

        self.archive_indv = np.empty((self.res, self.res), dtype=object)
        self.archive_fit  = np.full((self.res, self.res), -np.inf)
        # descriptor minmax; mean ang vel 0-3+, thrust saturation 0-1
        self.xrange = (0, 3)
        self.yrange = (0, 1)

        self.arms: dict['str', Callable] = {
            'random': arms.random,
            'gaussian': arms.gaussian
        }
        self.bandit = MAB(list(self.arms.keys()))

    def propose(self, qty, Mpool=None) -> tuple[ list[Individual], dict ]:
        # initial bootsrap
        if self.gen == 0:
            proposition: list[Individual] = self.arms['random'](None, None, qty)
            return proposition, {'budget': {'random': qty}}

        # get the arm budget
        budget = self.bandit.pull(qty)
        proposition: list[Individual] = []

        # for each arm add its budgeted propositions
        for arm, budg in budget.items():
            indvs = self.arms[arm](self.archive_indv, self.archive_fit, budg)
            proposition.extend(indvs)

        return proposition, {'budget': dict(budget)}


    def update(self, individuals: list[Individual]):
        bandit_score = {k: 0.0 for k in self.arms.keys()}

        stats = {'updates': 0, 'bandit_score': bandit_score}
        for i in individuals:
            # calculate the archive coordinates for individual
            xval, yval = i.descriptors['ang_vel'], i.descriptors['saturation']
            idx = int(((xval - self.xrange[0]) / (self.xrange[1] - self.xrange[0])) * self.res)
            idx = np.clip(idx, 0, self.res - 1) # keep index in bounds

            idy = int(((yval - self.yrange[0]) / (self.yrange[1] - self.yrange[0])) * self.res)
            idy = np.clip(idy, 0, self.res - 1) # keep index in bounds

            # if slot empty fill it and reward bandit:
            if i.fitness >= self.archive_fit[idx, idy]:
                stats['updates'] += 1
            else:
                continue

            # update the archives
            self.archive_fit[idx, idy]  = i.fitness
            self.archive_indv[idx, idy] = i
            bandit_score[i.tag] += 1

        # update the bandit
        if self.gen > 0:
            self.bandit.update_stats(bandit_score)
        if self.gen > 99:
            self.bandit.decay = 0.99

        # smoke-test diagnostics: is the archive actually filling?
        occupied = self.archive_fit > -np.inf
        stats['coverage']     = int(occupied.sum()) / (self.res * self.res)
        stats['archive_best'] = float(self.archive_fit[occupied].max()) if occupied.any() else None

        # increment gen
        self.gen += 1
        return stats

    def revalidate(self, individual):
        pass

if __name__ == "__main__":
    import time
    from modules.simulation import sim1
    alg = algorithm(50)

    max_seconds = 120
    batch_size = 2000
    start = time.time()
    i = 0
    trajectory = []  # (elapsed, archive_best, top10_mean) per gen
    total_evals = 0
    while time.time() - start < max_seconds:
        indv, propstat = alg.propose(batch_size)
        simstat        = sim1.sim(indv, {'limit': 10, 'length': 10}, seed=42)
        updatestat     = alg.update(indv)
        elapsed = time.time() - start
        total_evals += batch_size

        occ = alg.archive_fit[alg.archive_fit > -np.inf]
        top10 = np.sort(occ)[-10:].mean() if len(occ) >= 10 else occ.mean()
        trajectory.append((elapsed, updatestat['archive_best'], top10))

        print(f"gen {i} [{elapsed:.1f}s]:\n\t budget={propstat['budget']}\n\t fit_mean={simstat['fit_mean']:.3f} fit_max={simstat['fit_max']:.3f}\n\t updates={updatestat['updates']}\n\t coverage={updatestat['coverage']:.2f} archive_best={updatestat['archive_best']:.3f}\n\t bandit_score={updatestat['bandit_score']}")
        i += 1

    occupied = alg.archive_fit[alg.archive_fit > -np.inf]
    times = np.array([t[0] for t in trajectory])
    bests = np.array([t[1] for t in trajectory])
    top10s = np.array([t[2] for t in trajectory])

    # AUC of archive_best over time (variance-robust: integrates whole trajectory)
    auc_best  = np.trapz(bests,  times) / times[-1]
    auc_top10 = np.trapz(top10s, times) / times[-1]

    # time-to-threshold (when did archive_best first cross X?)
    def time_to(threshold):
        idx = np.argmax(bests >= threshold)
        return times[idx] if bests[idx] >= threshold else None

    print(f"\n--- batch_size={batch_size} comparison metrics ---")
    print(f"  generations:     {i}")
    print(f"  total evals:     {total_evals}")
    print(f"  evals/sec:       {total_evals / times[-1]:.0f}")
    print(f"  final top10:     {top10s[-1]:.3f}   (more stable than max)")
    print(f"  AUC archive_best:{auc_best:.3f}     (time-avg best, integrates whole run)")
    print(f"  AUC top10:       {auc_top10:.3f}     (time-avg top10, smoothest signal)")
    for thr in [2.0, 3.0, 4.0, 5.0]:
        t = time_to(thr)
        print(f"  time to fit>={thr}: {f'{t:.1f}s' if t is not None else 'not reached'}")

    import matplotlib.pyplot as plt
    display = np.where(alg.archive_fit > -np.inf, alg.archive_fit, np.nan)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(display.T, origin='lower', aspect='auto', cmap='viridis',
                   extent=[alg.xrange[0], alg.xrange[1], alg.yrange[0], alg.yrange[1]])
    plt.colorbar(im, ax=ax, label='fitness')
    ax.set_xlabel('mean |angular velocity| (rad/s)')
    ax.set_ylabel('mean thrust saturation')
    ax.set_title(f'archive fitness — gen {alg.gen}')
    plt.tight_layout()
    plt.savefig('archive_heatmap.png', dpi=150)
    plt.show()
