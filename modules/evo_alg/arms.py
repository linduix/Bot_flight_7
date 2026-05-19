from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.evo_alg.mapElites import Archive

from modules.individual import Individual
import numpy as np
import tomllib

def random(archive: Archive, qty) -> list[Individual]:
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

        child   = Individual('random', weights=weights, biases=biases, parent_idx=None)
        children.append(child)

    return children

def gaussian(archive: Archive, qty) -> list[Individual]:
    std = 0.1
    rng = np.random.default_rng()

    # weight by curiosity (for exploration)
    w = np.where(archive.fit != -np.inf, archive.curi, 0) # 0 weight for empty cells
    assert(w.sum() > 0)
    probs = w.ravel() / w.sum()

    # randomly choose idx from occupied coordinates
    choices = rng.choice(len(probs), size=qty, p=probs)
    row, col = np.unravel_index(choices, shape=w.shape)

    children = []
    for i, j in zip(row, col):
        indv: Individual = archive.get(i, j)
        # mutate weights
        weights = indv.weights + rng.standard_normal(size=indv.weights.size) * std
        biases  = indv.biases  + rng.standard_normal(size=indv.biases.size ) * std
        # make child
        child   = Individual('gaussian', weights=weights, biases=biases, parent_idx=(i, j))
        children.append(child)

    return(children)

def iso(archive: Archive, qty) -> list[Individual]:
    rng = np.random.default_rng()
    children = []

    # calculate weights by improvement (for exploitation)
    w = np.where(archive.fit != -np.inf, archive.impr, 0) # 0 weight for empty cell
    assert(w.sum() > 0)
    probs = w.ravel() / w.sum() # flatened matrix

    chosen_idx = rng.choice(probs.size, size=qty, p=probs)   # choose idx based on weight
    pa_r, pa_c = np.unravel_index(chosen_idx, shape=w.shape) # turn flat idx to matrix coords

    # get coords of all occupied
    occupied = np.argwhere(archive.fit > -np.inf)

    for row, col in zip(pa_r, pa_c):
        # get parent a
        pa: Individual = archive.get(row, col)
        # get the distances from parent a
        diff = occupied - np.array([row, col])
        dist = np.linalg.norm(diff, axis=1)
        self_mask = dist == 0

        # weight the rest by distance, closer is better
        b_weights = 1 / (1 + dist ** 1.2)
        b_weights[self_mask] = 0
        assert(b_weights.sum() > 0)
        b_probs   = b_weights / b_weights.sum()

        # get parent b
        chosen_idx = rng.choice(b_probs.size, p=b_probs)
        pb_r, pb_c = occupied[chosen_idx]
        pb: Individual = archive.get(pb_r, pb_c)

        # child weight = pa weight + noise + lerp between parent a and b
        noise_strength = 0.05
        lerp_strength = 0.1
        t = rng.standard_normal()
        w_noise = rng.standard_normal(size=pa.weights.size) * noise_strength
        b_noise = rng.standard_normal(size=pa.biases.size ) * noise_strength

        # comence lerping
        child_w = pa.weights + w_noise + (pb.weights - pa.weights) * t * lerp_strength
        child_b = pa.biases  + b_noise + (pb.biases  - pa.biases ) * t * lerp_strength

        child = Individual('iso', weights=child_w, biases=child_b, parent_idx=(row, col))

        children.append(child)

    return children
