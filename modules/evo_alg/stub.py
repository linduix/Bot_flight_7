from modules.individual import Individual
import numpy as np

class evostub:
    def propose(self, qty, Mpool):
        individuals = []
        for _ in range(qty):
            weights = np.array([[0.0, 0.0]])
            indv = Individual(weights=weights, tag='stub')
            individuals.append(indv)

        stats = {'propose': 'success'}
        return individuals, stats

    def update(self, individuals):
        stats = {'update': 'success'}
        return stats
