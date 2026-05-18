from modules.individual import Individual
from modules.evo_alg import arms
import numpy as np

class algorithm():
    def __init__(self, resolution) -> None:
        self.res = resolution

        self.archive_indv = np.empty((self.res, self.res), dtype=object)
        self.archive_fit  = np.full((self.res, self.res), -np.inf)
        # descriptor minmax; mean ang vel 0-10+, thrust saturation 0-1
        self.xrange = (0, 10)
        self.yrange = (0, 1)

        self.arms = {'random': arms.random}

        pass
    def propose(self, qty, Mpool=None):
        return

    def update(self, individuals: list[Individual]):
        bandit_score = {k: 0 for k in self.arms.keys()}

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

            self.archive_fit[idx, idy]  = i.fitness
            self.archive_indv[idx, idy] = i
            bandit_score[i.tag] += 1

        return stats

    def revalidate(self, individual):
        pass
