from modules.individual import Individual
import numpy as np
import tomllib

config_path = 'config.toml'

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

        x = obs # ( n , observations, k ) n = drones, k = observations per drone
        for W, b in zip(self.weights, self.biases):
            x = np.einsum('noi,nik->nok',W, x)         # (n, out, in) @ (n, in, k)
            x = x + b[:, :, np.newaxis] # (n, out, k) + (n, bias, 1) added 3d axis to bias
            x = np.tanh(x)

        return(x) # output (n, output, k)  
    

