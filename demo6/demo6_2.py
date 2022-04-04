import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i+h, j: j+w] * K).sum()
    return Y


class Conv2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])


def corr2d_multi_in_out_1x1(X, K):
    ci, h, w = X.shape
    co = K.shape[0]
    X = X.reshape((ci, h*w))
    K = K.reshape((co, ci))
    Y = torch.matmul(K, X)
    return Y.reshape((co, h, w))


# X = torch.normal(0, 1, (3, 3, 3))
# K = torch.normal(0, 1, (2, 3, 1, 1))
# Y = corr2d_multi_in_out_1x1(X, K)
# print(K.shape)


conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)
