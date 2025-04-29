import torch.nn as nn

class FashionMNISTANN(nn.Module):
    def __init__(self, input_size=784, output_size=10, hidden_layers=2, hidden_nodes=256,
                 activation='relu', dropout=0.0, output_activation='none'):
        super(FashionMNISTANN, self).__init__()

        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()

        # Activation mapping
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU()
        }
        act_fn = activations[activation]

        # Build hidden layers
        self.layers.append(nn.Linear(input_size, hidden_nodes))
        self.layers.append(act_fn)
        if dropout > 0:
            self.layers.append(nn.Dropout(dropout))

        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_nodes, hidden_nodes))
            self.layers.append(act_fn)
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))

        # Final output layer
        self.layers.append(nn.Linear(hidden_nodes, output_size))

        # Output activation (for NLLLoss)
        output_activations = {
            'none': nn.Identity(),          # Do nothing
            'logsoftmax': nn.LogSoftmax(dim=1)  # Needed for NLLLoss
        }
        self.output_activation = output_activations[output_activation]

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_activation(x)
        return x
