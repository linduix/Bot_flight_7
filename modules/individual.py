from dataclasses import dataclass
import numpy as np

@dataclass
class Individual:
    tag:         str
    weights:     np.ndarray
    fitness:     float|None = None
    descriptors: tuple|None = None
    stats:       list|None  = None
