from dataclasses import dataclass, field
import numpy as np

@dataclass
class Individual:
    tag:         str
    weights:     np.ndarray
    biases:      np.ndarray
    parent_idx:  tuple[int, int]|None
    fitness:     float|None = None
    descriptors: dict       = field(default_factory=dict)
    stats:       list|None  = None
    improv:      float|None = None
