from dataclasses import dataclass
import numpy as np

@dataclass
class Individual:
    tag:         str
    weights:     np.ndarray
    biases:      np.ndarray
    fitness:     float|None = None
    descriptors: dict|None = None
    stats:       list|None  = None
