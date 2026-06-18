import torch.nn as nn
import torch
from utils import set_random_seed


class Header(nn.Module):
    def __init__(self, num_channels=6):
        super(Header, self).__init__()
        torch.manual_seed(5)
        self.header = nn.Conv2d(in_channels=num_channels, out_channels=3, kernel_size=1, bias=False)

    @torch.no_grad()
    def forward(self, x):
        s0, s1 = x.size(0), x.size(1)
        x = x.contiguous().view(s0 * s1, x.size(2), x.size(3), x.size(4))
        x = self.header(x)
        # out = x
        out = x.contiguous().view(s0, s1, x.size(1), x.size(2), x.size(3))
        return out


class SimpleHeader(nn.Module):
    def __init__(self, num_channels=6):
        super(SimpleHeader, self).__init__()
        set_random_seed(55)
        self.header = nn.Conv2d(in_channels=num_channels, out_channels=3, kernel_size=1, bias=False)

    @torch.no_grad()
    def forward(self, x):
        out = self.header(x)
        return out


if __name__ == "__main__":
    import torch

    input = torch.randn(10, 5, 6, 84, 84)
    model = SimpleHeader(num_channels=6)
    print(model)
    output = model(input)
    print(output.size())

