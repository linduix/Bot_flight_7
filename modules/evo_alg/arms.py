from __future__ import annotations
from typing import TYPE_CHECKING
from cmaes import SepCMA
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

# Explorer
def gaussian(archive: Archive, qty) -> list[Individual]:
    std = 0.3
    rng = np.random.default_rng()

    # weight by curiosity (for exploration)
    # using softmax formula
    Temp = 0.9
    w = archive.curi
    w_stable = w - w.max()
    probs = np.exp(w_stable / Temp).ravel()
    probs[archive.fit.ravel() == -np.inf] = 0.0
    assert(probs.sum() > 0)
    probs /= probs.sum()

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

# Exploiter
def iso(archive: Archive, qty) -> list[Individual]:
    rng = np.random.default_rng()
    children = []

    # calculate weights by improvement (for exploitation)
    Temp = 0.9
    w = archive.impr
    w_stable = w - w.max()
    probs = np.exp(w_stable / Temp).ravel()
    probs[archive.fit.ravel() == -np.inf] = 0.0
    assert(probs.sum() > 0)
    probs /= probs.sum()

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
        noise_strength = 0.1
        lerp_strength = 0.4
        t = rng.standard_normal()
        w_noise = rng.standard_normal(size=pa.weights.size) * noise_strength
        b_noise = rng.standard_normal(size=pa.biases.size ) * noise_strength

        # comence lerping
        child_w = pa.weights + w_noise + (pb.weights - pa.weights) * t * lerp_strength
        child_b = pa.biases  + b_noise + (pb.biases  - pa.biases ) * t * lerp_strength

        child = Individual('iso', weights=child_w, biases=child_b, parent_idx=(row, col))

        children.append(child)

    return children

class cma():
    def __init__(self, sigma=0.1) -> None:
        self.sigma = sigma

        # dummy variables
        self.n_weights = 0
        self.cma     = SepCMA(mean=np.zeros(2), sigma=sigma)
        self.pop     = np.inf

        self.buffer  = []
        self.best    = -np.inf
        self.stag    = 0
        self.restart = True


    def ask(self, archive: Archive, qty) -> list[Individual]:
        if self.restart:
            self._reset(archive)
            self.restart = False

        children = []
        for _ in range(qty):
            g = self.cma.ask()
            w = g[:self.n_weights].astype(np.float32)
            b = g[self.n_weights:].astype(np.float32)
            child = Individual('cma', w, b, None)
            children.append(child)

        return children
    
    def tell(self, individuals: list[Individual]):
        filterd = [i for i in individuals if i.tag == 'cma']
        for i in filterd:
            g = np.concatenate([i.weights, i.biases])
            assert i.fitness is not None, 'fitness is None in cma.tell'
            self.buffer.append((g, i.fitness))

        while len(self.buffer) >= self.pop:
            batch, self.buffer = self.buffer[:self.pop] , self.buffer[self.pop:]
            self.cma.tell([(g, -f) for g, f in batch])

            best = max(f for _, f in batch)
            if best > self.best + 1e-4:
                self.best = best
                self.stag = 0
            else:
                self.stag += 1

        if self.cma.should_stop() or self.stag > 15:
            self.restart = True

    def _reset(self, archive: Archive):
        # calculate weights by improvement (for exploitation)
        Temp = 0.9
        w = archive.fit.copy()
        w_stable = w - w[np.isfinite(w)].max()
        probs = np.exp(w_stable / Temp).ravel()
        probs[archive.fit.ravel() == -np.inf] = 0.0
        assert probs.sum() > 0
        probs /= probs.sum()

        chosen_idx = np.random.choice(probs.size, p=probs)   # choose idx based on weight
        r, c = np.unravel_index(chosen_idx, shape=w.shape) # turn flat idx to matrix coord
        chosen: Individual = archive.indv[r, c] # type: ignore

        # create new cma instance
        base = np.concat([chosen.weights, chosen.biases])
        self.cma    = SepCMA(mean=base, sigma=self.sigma)
        self.best   = -np.inf
        self.buffer = []
        self.stag   = 0
        self.pop    = self.cma.population_size
        self.n_weights = len(chosen.weights)
