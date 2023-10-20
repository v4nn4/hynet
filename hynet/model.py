import logging

import torch
from torch import nn

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
torch.manual_seed(1337)


class LeNet(nn.Module):
    """LeNet-5 (1998)"""

    def __init__(self, N: int, C: int, mean: float = 0.5, std: float = 0.01):
        super(LeNet, self).__init__()
        self.mean = mean
        self.std = std
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.D = int(((N - 4) / 2 - 4) / 2)  # 4 for N = 28
        self.fc1 = nn.Linear(16 * self.D * self.D, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, C)
        self.sm = nn.Softmax(dim=1)

        # Initialize weights
        gain = nn.init.calculate_gain("tanh")
        nn.init.xavier_normal_(self.conv1.weight, gain)
        nn.init.xavier_normal_(self.conv2.weight, gain)
        nn.init.xavier_normal_(self.fc1.weight, gain)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight, gain)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_normal_(self.fc3.weight, 1.0)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = (x - self.mean) / self.std  # normalize
        x = self.pool(torch.tanh(self.conv1(x)))  # conv2d + tanh + pool
        x = self.pool(torch.tanh(self.conv2(x)))  # conv2d + tanh + pool
        x = x.view(-1, 16 * self.D * self.D)  # flatten
        x = torch.tanh(self.fc1(x))  # fc layer + tanh
        x = torch.tanh(self.fc2(x))  # fc layer + tanh
        x = self.fc3(x)  # fc layer
        return self.sm(x)  # softmax to output probabilities
