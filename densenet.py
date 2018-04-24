import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def initializer(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform(param.data)
        elif 'bias' in name:
            param.data.zero_()
        elif param.dim() == 1:
            nn.init.xavier_uniform(param.data)
        else:
            nn.init.xavier_uniform(param.data)

class BottleneckSingleLayer(nn.Module):
    def __init__(self, input_size, growth_rate):
        super().__init__()
        hidden_size = 4 * input_size
        self.bn1 = nn.BatchNorm2d(input_size)
        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))    # BU - ReLU - Conv(1*1)
        out = self.conv2(F.relu(self.bn2(out)))  # BU - ReLU - Conv(3*3)
        out = torch.cat((x, out), dim=1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, input_size, growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm2d(input_size)
        self.conv = nn.Conv2d(input_size, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = torch.cat((x, out), dim=1)
        return out

class Transition(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.bn = nn.BatchNorm2d(input_size)
        self.conv = nn.Conv2d(input_size, output_size, kernel_size=1)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

