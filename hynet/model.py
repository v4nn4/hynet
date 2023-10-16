import logging

import torch
from torch import nn

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
torch.manual_seed(1337)


class LeNet(nn.Module):
    """LeNet-5 (1998)"""

    def __init__(self, N: int, C: int):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_bn = nn.BatchNorm2d(16)
        # image dimension reduces with convolution kernels and pooling
        self.D = int(((N - 4) / 2 - 4) / 2)  # 4 for N = 28
        self.fc1 = nn.Linear(16 * self.D * self.D, 120)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc2_bn = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, C)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2_bn(self.conv2(x))))
        x = x.view(-1, 16 * self.D * self.D)  # flatten
        x = torch.tanh(self.fc1_bn(self.fc1(x)))
        x = torch.tanh(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return self.sm(x)


def initialize_weights(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("tanh"))
