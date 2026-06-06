from modules.individual import Individual
import numpy as np
import tomllib

config_path = 'config.toml'

def leaky_relu(x: np.ndarray):
    return np.maximum(x, x * 0.01)

def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


class brain():
    def __init__(self, individuals: list[Individual]) -> None:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
        # combine all drones into one 3d matrix for batch forwarding
        self.weights = []
        self.biases =  []
        shape = config['network']['layers']

        weight_start = 0
        bias_start   = 0
        for i in range(len(shape)-1):
            # get the shape info
            inp = shape[i]
            out = shape[i+1]
            conns = inp * out

            # pointers to help slice flat weights/bias array
            weight_end = weight_start + conns
            bias_end   = bias_start   + out

            # reshape layer i for every individual
            shaped_w = [np.reshape(ind.weights[weight_start:weight_end], (out, inp)) for ind in individuals]
            shaped_b = [np.reshape(ind.biases[bias_start:bias_end], (out))           for ind in individuals]

            # combine to 3d matrix for layer i
            self.weights.append(np.stack(shaped_w))
            self.biases.append( np.stack(shaped_b))

            # move pointer
            weight_start = weight_end
            bias_start = bias_end

    def forward(self, obs: np.ndarray, alive: np.ndarray | None = None, prev_actions: np.ndarray | None = None):
        """Forward pass over a batched obs tensor.

        obs:          (N, in, S)   observations per (drone, seed) pair
        alive:        (N,) bool    optional per-drone alive mask. when given and not all
                                   True, only alive drones are processed; dead drones keep
                                   their `prev_actions` outputs (or zero if not provided).
        prev_actions: (N, out, S)  last tick's actions, used as the kept output for dead drones.
        """
        assert obs.ndim == 3, f"obs must be 3D (n, observations, k), got shape {obs.shape}"
        layers = len(self.weights)

        # if any drone is fully dead, run the forward pass on the alive subset only.
        # the weights are (N, out, in), biases (N, out) — slicing along axis 0 gives the
        # smaller batch in one copy, then the matmul handles the reduced N naturally.
        if alive is not None and not alive.all():
            x = obs[alive]                              # (N_alive, in, S)
            ws = [W[alive] for W in self.weights]
            bs = [b[alive] for b in self.biases]
            for i, (W, b) in enumerate(zip(ws, bs)):
                x = W @ x + b[:, :, np.newaxis]
                if i < layers - 1:
                    x = leaky_relu(x)
                else:
                    x[:, :2, :] = sigmoid(x[:, :2, :])
                    x[:, 2:, :] = np.tanh(x[:, 2:, :])
            # scatter back into a full (N, out, S) tensor; dead drones get prev_actions (or 0)
            if prev_actions is not None:
                full = prev_actions.copy()
            else:
                full = np.zeros((obs.shape[0], x.shape[1], obs.shape[2]), dtype=x.dtype)
            full[alive] = x
            return full

        # full-batch path (all drones alive, or no alive mask given)
        x = obs # ( n , observations, k ) n = drones, k = observations per drone
        for i,(W, b) in enumerate(zip(self.weights, self.biases)):
            x = W @ x                                  # (n, out, in) @ (n, in, k)
            x = x + b[:, :, np.newaxis] # (n, out, k) + (n, bias, 1) added 3d axis to bias
            if i < layers-1:
                x = leaky_relu(x)
            else:
                x[:, :2, :] = sigmoid(x[:, :2, :]) # 1-2 sigmoid
                x[:, 2:, :] = np.tanh(x[:, 2:, :]) # 3-4 tanh

        # thrust1, thrust2, rot1, rot2
        return(x) # output (n, output, k)
