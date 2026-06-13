import torch.nn as nn
import math


# --- gaussian initialize ---
def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


# --- Convolution block ---
class ConvBlock(nn.Module):
    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim

        self.C = nn.Conv1d(indim, outdim, 3, padding=padding)
        self.BN = nn.BatchNorm1d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool1d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)
        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


# --- flatten tensor ---
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


# --- avg pool ---
class AvgPool(nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return self.avgpool(x)


# --- ConvNet module ---
class ConvNet(nn.Module):
    def __init__(self, indim, depth, avgpool=True, flatten=True):
        super(ConvNet, self).__init__()
        self.grads = []
        self.fmaps = []
        trunk = []
        for i in range(depth):
            indim = indim if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for fist 4 layers
            trunk.append(B)

        if avgpool:
            trunk.append(AvgPool())

        if flatten:
            trunk.append(Flatten())


        self.trunk = nn.Sequential(*trunk)
        # size of flatten features for input image size of 3*32*32
        self.out_features = 64

    def forward(self, x):
        out = self.trunk(x)
        return out


def Conv4(indim=3, **kwargs):
    return ConvNet(indim, 4)


if __name__ == "__main__":
    import torch

    conv = Conv4(1)
    # summary(conv, (3,32,32), device='cpu')
    input = torch.randn(8, 5, 56)

    out = conv(input)

    print(out.shape)
