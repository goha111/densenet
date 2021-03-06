import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def initializer(m):
    for name, param in m.named_parameters():
        if 'weight' in name and param.dim() > 1:
            nn.init.xavier_uniform(param.data)
        elif 'bias' in name:
            param.data.zero_()
        elif param.dim() > 1:
            nn.init.xavier_uniform(param.data)
        else:
            param.data.zero_()

class BottleneckSingleLayer(nn.Module):
    def __init__(self, input_size, growth_rate):
        super().__init__()
        hidden_size = 4 * growth_rate
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

class DenseBlock(nn.Module):
    def __init__(self, input_size, growth_rate, num_layers=5, bottleneck=True):
        super().__init__()
        layers = []
        in_channel = input_size
        for i in range(num_layers):
            if bottleneck:
                layers.append(BottleneckSingleLayer(in_channel, growth_rate))
            else:
                layers.append(SingleLayer(in_channel, growth_rate))
            in_channel += growth_rate
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)

class DenseNet(nn.Module):
    def __init__(self, growth_rate, depth, reduction, nClasses, bottleneck=True):
        super().__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1)
        self.dense1 = DenseBlock(nChannels, growth_rate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growth_rate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = DenseBlock(nChannels, growth_rate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growth_rate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = DenseBlock(nChannels, growth_rate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growth_rate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), out.shape[2]))
        out = F.log_softmax(self.fc(out))
        return out
