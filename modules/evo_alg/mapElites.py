from modules.individual import Individual
from modules.evo_alg import arms
from typing import cast
import pickle as pkl
import numpy as np
import os

class MAB():
    def __init__(self, arms: list[str]) -> None:
        self.decay = 0.99
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
            exploration = np.sqrt(0.1 * np.log(self.total_pulls) / stats['pulls'])
            arm_value   = mean_score + exploration

            stats['value'] = arm_value

    def pull(self, qty: int) -> list[str]: # dict[str, int]:
        budget = {k: 0 for k in self.arms}
        pulls = []

        for _ in range(qty):
            self.recompute_values()
            # get for arm with most value
            best = max(self.arms, key=lambda k: self.arms[k]['value'])

            # increment the budget + pulls
            budget[best]     += 1
            pulls.append(best)
            self.total_pulls += 1
            self.arms[best]['pulls'] += 1

        return pulls

class Archive():
    def __init__(self, res) -> None:
        # descriptor minmax; x: mean gimble angle, y: activation variance
        self.xrange: tuple = (0.20, .8)
        self.yrange: tuple = (0.02, 0.25)

        # archive matrices
        self.res  = res
        self.indv:    np.ndarray = np.empty((res, res), dtype=object) # matrix of Individuals
        self.fit :    np.ndarray = np.full((res, res),  -np.inf)      # fitness matrix
        self.curi:    np.ndarray = np.full((res, res),  0.01000)      # curiosity matrix
        self.impr:    np.ndarray = np.full((res, res),  0.01000)      # fitness improvement matrix
        self.failed:    np.ndarray = np.full((res, res), 0)            # failed-insert count per cell
        self.successes: np.ndarray = np.full((res, res), 0)            # successful-insert count per cell (discovery + improvement)

        self.curi_decay = 0.99

    def __setstate__(self, state):
        # compat: pre-failure-rate checkpoints lack failed/successes matrices.
        # backfill with zeros so resumed runs don't AttributeError.
        self.__dict__.update(state)
        if 'failed' not in state:
            self.failed = np.zeros((self.res, self.res), dtype=int)
        if 'successes' not in state:
            self.successes = np.zeros((self.res, self.res), dtype=int)

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
            self.failed[idx, idy] += 1
            if i.parent_idx is not None:
                self.curi[i.parent_idx] = max(self.curi[i.parent_idx] * self.curi_decay, 0.01)
            return cast(float, i.fitness - old_fit)

        # update the archivess
        self.fit[idx, idy]  =  i.fitness
        self.indv[idx, idy] =  i
        self.successes[idx, idy] += 1
        if i.parent_idx is not None:
            self.curi[i.parent_idx] += 1
        if old_fit != -np.inf:
            self.impr[idx, idy] += i.fitness - old_fit

        return cast(float, i.fitness - old_fit if old_fit != -np.inf else i.fitness)

    def get(self, row: int, col: int) -> Individual:
        return cast(Individual, self.indv[row, col])

    def pop(self) -> list[Individual]:
        indv = [x for x in self.indv.flat if x is not None]
        return cast(list[Individual], indv)

class algorithm():
    def __init__(self, resolution) -> None:
        self.gen = 0
        self.archive = Archive(resolution)

        # Bench rotation bandit design
        self.emitter_batch_size = 50
        self.max_active = 0  # derived from population in propose()
        self.active: list[object] = []
        self.stateful = {'cma_improv', 'cma_fit'}
        self.bench = {
            'cma_improv': arms.cma_improv,
            'cma_fit'   : arms.cma_fit,
            'random'    : arms.random,
            'gaussian'  : arms.gaussian,
            'iso'       : arms.iso,
        }

        self.bandit = MAB(list(self.bench.keys()))

    def propose(self, qty, Mpool=None) -> tuple[ list[Individual], dict ]:
        # initial bootsrap
        self.batch_size = qty
        if self.gen == 0:
            proposition: list[Individual] = arms.random(self.archive, qty)
            return proposition, {'active': 1, 'random': 1}

        # cap active emitters so total proposals (max_active * emitter_batch_size)
        # stay within the population budget
        self.max_active = qty // self.emitter_batch_size

        # evict dead stateful emitters, freeing slots for the bandit to refill
        self.active = [inst for inst in self.active if not getattr(inst, 'kill', False)]

        # spawn new emitters to fill vacant slots
        qty_short = self.max_active - len(self.active)
        pulls = self.bandit.pull(qty_short)
        proposition: list[Individual] = []

        for pull in pulls:
            if pull in self.stateful:
                instance = self.bench[pull]()
            else:
                instance = arms.stateless_wrapper(self.bench[pull])
            self.active.append(instance)

        for i, instance in enumerate(self.active):
            indvs = instance.ask(self.archive, self.emitter_batch_size)
            for ind in indvs:
                ind.instance = i
            proposition.extend(indvs)

        counts: dict[str, int] = {}
        for inst in self.active:
            counts[inst.tag] = counts.get(inst.tag, 0) + 1
        return proposition, {'active': len(self.active), **counts}


    def update(self, individuals: list[Individual]):
        bandit_score = {k: 0.0 for k in self.bench.keys()}
        self.archive.curi_decay = 0.1 ** (8/self.batch_size) # curiosity decay factor
        instance_indvs    = {i: [] for i, instance in enumerate(self.active) if instance.stateful}
        instance_updates  = {i: 0  for i in instance_indvs}  # archive inserts per stateful instance this gen

        stats = {'discoveries': 0, 'updates': 0, 'bandit_score': bandit_score}
        max_contrib = 0.0  # biggest single-individual contribution this batch
        for i in individuals:
            idx, idy = self.archive.coordinates(i)
            was_empty = self.archive.fit[idx, idy] == -np.inf
            delta = self.archive.insert(i)
            i.improv = delta
            delta = max(delta, 0.0)

            # if successful update, reward bandit:
            if delta > 0:
                if was_empty:
                    stats['discoveries'] += 1
                else:
                    stats['updates'] += 1

                if i.tag in bandit_score:
                    old_fit = i.fitness - delta                   # type:ignore
                    if old_fit < 0:
                        # occupant is negative: shift both up by -old_fit so the
                        # occupant sits at 0 and the candidate sits at its improvement.
                        # keeps contrib positive + monotonic while climbing out of negatives.
                        contrib = (i.fitness - old_fit)**2        # type:ignore
                    else:
                        contrib = i.fitness**2 - old_fit**2       # type:ignore
                    bandit_score[i.tag] += 1/self.emitter_batch_size
                    max_contrib = max(max_contrib, contrib)

            # give feedback to any stateful instances
            if i.tag in self.stateful:
                assert isinstance(i.instance, int)
                instance_indvs[i.instance].append(i)
                if delta > 0:
                    instance_updates[i.instance] += 1


        for i, indvs in instance_indvs.items():
            self.active[i].tell(indvs, self.archive)

        # kill stateful emitters that either converged (should_stop) or have gone
        # >= 10 consecutive gens without a single archive update
        killed_counts: dict[str, int] = {}
        next_active: list = []
        for idx, inst in enumerate(self.active):
            if not inst.stateful:
                continue
            if inst.kill:
                killed_counts[inst.tag] = killed_counts.get(inst.tag, 0) + 1
                continue
            if instance_updates.get(idx, 0) > 0:
                inst.stagnation_count = 0
                next_active.append(inst)
            else:
                inst.stagnation_count = getattr(inst, 'stagnation_count', 0) + 1
                if inst.stagnation_count >= 10:
                    killed_counts[inst.tag] = killed_counts.get(inst.tag, 0) + 1
                else:
                    next_active.append(inst)

        if killed_counts:
            parts = ' '.join(f'{tag} {n}x' for tag, n in sorted(killed_counts.items()))
            print(f'[emitter kill] killed: {parts}', flush=True)

        self.active = next_active

        # normalize each individual's reward to [0, 1] by the batch's single
        # best contribution, NOT by the arm totals. dividing by max_contrib (one
        # individual) keeps each arm's score a sum of per-pull rewards, so the
        # MAB's later score/pulls lands in [0, 1] instead of being squashed by
        # the batch budget. the unchanged fitness**2 formula still makes
        # high-fitness refinement worth more, and volume is preserved.
        # if max_contrib > 0:
        #     for k in bandit_score:
        #         bandit_score[k] /= max_contrib

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

        self.bandit      = MAB(list(self.bench.keys()))
        # keep the CMA arms across curriculum transitions: their mean+covariance
        # live in genome space (network topology is unchanged between stages), so
        # the learned distribution is still valid progress. rebuilding them here
        # would discard it and force a cold re-seed every transition.

        for i in initial_pop:
            i.parent_idx = None
            self.archive.insert(i)

def save(path, alg, settings, seed):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pkl.dump({'archive': alg.archive, 'gen': alg.gen, 'settings': settings, 'seed': seed}, f)
    os.replace(tmp, path)

def load(path) -> tuple:
    with open(path, 'rb') as f:
        data = pkl.load(f)
    # new checkpoints store only the archive + gen; old ones store the whole alg
    archive = data['archive'] if 'archive' in data else data['alg'].archive
    gen     = data['gen']     if 'gen'     in data else data['alg'].gen
    return archive, gen, data['settings'], data['seed']

def load_alg(path) -> tuple:
    # reconstruct a fresh algorithm with the saved archive + gen restored, at the
    # archive's own resolution. for viz/diagnostic scripts that just want the
    # checkpoint's state without forcing a resolution.
    archive, gen, settings, seed = load(path)
    alg = algorithm(archive.res)
    alg.archive = archive
    alg.gen = gen
    return alg, settings, seed

import signal as _signal
def _worker_init():
    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
