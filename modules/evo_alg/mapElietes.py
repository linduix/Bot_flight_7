from modules.individual import Individual
from modules.evo_alg import arms
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
        # descriptor minmax; mean ang vel 0-10+, thrust saturation 0-1
        self.xrange = (0, 10)
        self.yrange = (0, 1)
        
        self.arms: dict['str', object] = {'random': arms.random}
        self.bandit = MAB(list(self.arms.keys()))

    def propose(self, qty, Mpool=None) -> tuple[ list[Individual], dict[str, str|int|float] ]:
        budget = self.bandit.pull(qty)
        proposition = []



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

        # increment gen
        self.gen += 1
        # update the bandit
        self.bandit.update_stats(bandit_score)
        if self.gen > 99:
            self.bandit.decay = 0.99
        return stats

    def revalidate(self, individual):
        pass
