import torch.nn as nn
import torch.nn.functional as F

class NeuralNetworks(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, saturate=False):
        """
        Initialize neural network
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(NeuralNetworks, self).__init__()

        self.h_nodes = hidden_dims
        self.saturate = saturate

        self.fc1 = nn.Linear(input_dim, self.h_nodes[0])
        # self.bn1 = nn.LayerNorm(self.h_nodes[0])
        self.fc2 = nn.Linear(self.h_nodes[0], self.h_nodes[1])
        # self.bn2 = nn.LayerNorm(self.h_nodes[1])
        self.fc3 = nn.Linear(self.h_nodes[1], self.h_nodes[2])
        # self.bn3 = nn.LayerNorm(self.h_nodes[2])
        self.fc4 = nn.Linear(self.h_nodes[2], output_dim)

        # Normalization: nn.BatchNorm1d(self.h_nodes[2])
        # Initialization: nn.init.xavier_normal_(self.fc1.weight.data)
        #                 nn.init.xavier_normal_(self.fc1.bias.data)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        if self.saturate:
            x = 2 * F.tanh(x)

        # x = F.leaky_relu(self.bn1(self.fc1(x)))
        # x = F.leaky_relu(self.bn2(self.fc2(x)))
        # x = F.leaky_relu(self.bn3(self.fc3(x)))
        # x = self.fc4(x)
        return x