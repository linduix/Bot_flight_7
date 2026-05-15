from modules.individual import Individual
import numpy as np
import tomllib

class evostub:
    def __init__(self) -> None:
        pass

    def propose(self, qty, Mpool) -> tuple[ list[Individual], dict[str, str|int|float] ]:
        individuals = []

        with open('config.toml', 'rb') as f:
            config = tomllib.load(f)

        total_weights = 0
        total_biases  = 0
        shape = config['network']['layers']
        for i in range(len(shape)-1):
            total_weights += shape[i] * shape[i+1]
            total_biases  += shape[i+1]

        for _ in range(qty):
            weights = np.random.randn(total_weights).astype(np.float32)
            biases  = np.random.randn(total_biases ).astype(np.float32)

            indv = Individual(weights=weights, biases=biases, tag='stub')
            individuals.append(indv)

        stats = {'propose': 'success', 'pop': qty, 'size': total_weights}
        return individuals, stats

    def update(self, individuals) -> dict:
        stats = {'update': 'success'}
        return stats
