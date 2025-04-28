import torch.nn as nn

class FashionMNISTANN(nn.Module):
    def __init__(self, layers=[784, 128, 64, 10], activation='relu', dropout=0.0):
        super(FashionMNISTANN, self).__init__()
        
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()

        # Map string to activation function
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU()
        }
        act_fn = activations[activation]

        # Build layers dynamically
        for i in range(len(layers) - 2):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            self.layers.append(act_fn)
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        
        # Final output layer (no activation)
        self.layers.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
        return x
