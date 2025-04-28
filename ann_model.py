import torch.nn as nn

class FashionMNISTANN(nn.Module):
    def __init__(self, input_size=784, output_size=10, hidden_layers=2, hidden_nodes=256, activation='relu', dropout=0.0):
        super(FashionMNISTANN, self).__init__()

        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()

        # Activation mapping
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU()
        }
        act_fn = activations[activation]

        # Input to first hidden layer
        self.layers.append(nn.Linear(input_size, hidden_nodes))
        self.layers.append(act_fn)
        if dropout > 0:
            self.layers.append(nn.Dropout(dropout))

        # Add additional hidden layers
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_nodes, hidden_nodes))
            self.layers.append(act_fn)
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))

        # Output layer
        self.layers.append(nn.Linear(hidden_nodes, output_size))

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
        return x
