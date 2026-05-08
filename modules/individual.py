from dataclasses import dataclass
import numpy as np

@dataclass
class Individual:
    weights = np.ndarray
    fitness = float
    descriptors = tuple
    stats = dict
    tag = str