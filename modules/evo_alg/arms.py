from matplotlib import colormaps
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

def iso(archive_indv: np.ndarray, archive_fit: np.ndarray, qty) -> list[Individual]:
    rng = np.random.default_rng()
    children = []

    # calculate weights
    w  = np.maximum(archive_fit, 0)
    assert(w.sum() > 0)
    probs = w.ravel() / w.sum()     # flatened matrix

    chosen_idx = rng.choice(probs.size, size=qty, p=probs)   # choose idx based on weight
    pa_r, pa_c = np.unravel_index(chosen_idx, shape=w.shape) # turn flat idx to matrix coords

    # get coords of all occupied
    occupied = np.argwhere(archive_fit > -np.inf)

    for row, col in zip(pa_r, pa_c):
        # get parent a
        pa: Individual = archive_indv[row, col]
        # get the distances from parent a
        diff = occupied - np.array([row, col])
        dist = np.linalg.norm(diff, axis=1)
        self_mask = dist == 0

        # weight the rest by distance, closer is better
        b_weights = 1 / (1 + dist ** 2)
        b_weights[self_mask] = 0
        assert(b_weights.sum() > 0)
        b_probs   = b_weights / b_weights.sum()

        # get parent b
        chosen_idx = rng.choice(b_probs.size, p=b_probs)
        pb_r, pb_c = occupied[chosen_idx]
        pb: Individual = archive_indv[pb_r, pb_c]

        # child weight = pa weight + noise + lerp between parent a and b
        size = pa.weights.size
        noise_strength = 0.15
        noise = rng.standard_normal(size=size) * noise_strength
        child = pa.weights + noise + (pb.weights - pa.weights) * rng.standard_normal(size=size)

        children.append(child)

    return children
