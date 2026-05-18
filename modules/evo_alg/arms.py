from modules.individual import Individual
import numpy as np
import tomllib

def random(archive_indv, archive_fit, qty) -> list[Individual]:
    children = []

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

        child   = Individual(weights=weights, biases=biases, tag='random')
        children.append(child)

    return children

def gaussian(archive_indv, archive_fit, qty) -> list[Individual]:
    std = 0.1

    rng = np.random.default_rng()
    # returns list of coordinates: eg 2d array; rows matches, columns coordinates x, y
    occupied = np.argwhere(archive_fit > -np.inf)

    # randomly choose idx from occupied coordinates
    choices = rng.choice(len(occupied), size=qty)
    children = []
    for c in choices:
        # get the coord pair from the idx
        i, j = occupied[c]

        indv: Individual = archive_indv[i, j]
        weights = indv.weights + rng.standard_normal(size=indv.weights.size) * std
        biases  = indv.biases  + rng.standard_normal(size=indv.biases.size ) * std
        child   = Individual('gaussian', weights=weights, biases=biases)
        children.append(child)

    return(children)