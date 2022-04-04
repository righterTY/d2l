import numpy as np
import time
import torch
from d2l import torch as d2l
from torch.utils import data


class Timer:
    """记录时间"""

    def __init__(self):
        self.tik = None
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        """累加求和"""
        return np.array(self.times).cumsum().tolist()


def synthetic_data(w, b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w)))  # len(w)为特征数
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    num_examples = len(features)  # 数据的数量
    indices = list(range(num_examples))
    np.random.shuffle(indices)  # 打乱数据编号顺序
    for i in range(0, num_examples, batch_size):  # 每次取batch_size个数据
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        # batch_indices = indices[i: min(i+batch_size, num_examples)]
        yield features[batch_indices], labels[batch_indices]  # 生成器，返回迭代器，只能用于迭代操作，例如for循环


def linreg(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


def sgd(params, lr, batch_size):  # @save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == '__main__':
    batch_size = 10
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss

    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    features, labels = synthetic_data(w, b, 100)

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')