import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层（Linear:全连接层）
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=True)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


class CombineModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(20, 32), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(20, 64), nn.Tanh())

    def forward(self, X):
        return torch.cat((self.net1(X), self.net2(X)), 1)


def block1():
    return nn.Sequential(nn.Linear(20, 10), nn.ReLU(),
                         nn.Linear(10, 20), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}', block1())
    return net


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


class CenterLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


class DiyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


X = torch.rand(2, 20)
# net = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 5))
# net(X)
# print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# print(net)
# print(*[(name, param.shape) for name, param in net[0][0][0].named_parameters()])
# net = NestMLP()
net = DiyLinear(20, 10)
# net.apply(init_constant)
net(X)
print(net)
print(*[(name, param.shape) for name, param in net.named_parameters()])
# print(*[(name, param.shape) for name, param in net.named_parameters()])