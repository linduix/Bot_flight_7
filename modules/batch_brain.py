from modules.individual import Individual
import numpy as np
import tomllib

config_path = 'config.toml'

def leaky_relu(x: np.ndarray):
    return np.maximum(x, x * 0.01)

def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


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

    def forward(self, obs: np.ndarray):
        assert obs.ndim == 3, f"obs must be 3D (n, observations, k), got shape {obs.shape}"
        layers = len(self.weights)
        x = obs # ( n , observations, k ) n = drones, k = observations per drone
        for i,(W, b) in enumerate(zip(self.weights, self.biases)):
            x = np.einsum('noi,nik->nok',W, x)         # (n, out, in) @ (n, in, k)
            x = x + b[:, :, np.newaxis] # (n, out, k) + (n, bias, 1) added 3d axis to bias
            if i < layers-1:
                x = leaky_relu(x)
            else:
                x[:, :2, :] = sigmoid(x[:, :2, :]) # 1-2 sigmoid
                x[:, 2:, :] = np.tanh(x[:, 2:, :]) # 3-4 tanh

        # thrust1, thrust2, rot1, rot2
        return(x) # output (n, output, k)
