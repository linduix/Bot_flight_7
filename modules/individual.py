from dataclasses import dataclass, field
import numpy as np

@dataclass
class Individual:
    tag:         str
    weights:     np.ndarray
    biases:      np.ndarray
    fitness:     float|None = None
    descriptors: dict       = field(default_factory=dict)
    stats:       list|None  = None
