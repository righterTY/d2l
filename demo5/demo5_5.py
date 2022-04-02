import torch
from torch import nn
from torch.nn import functional as F
import time


class Timer:
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


def try_gpu(i = 0):  # 查看可用gpu，并返回
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


timer = Timer()
timer.start()
f_norm = []
for i in range(1000):
    x = torch.randn(100, 100, device=try_gpu())
    y = torch.randn(100, 100, device=try_gpu())
    z = x * y
    f_norm.append(torch.norm(z, p='fro'))
print(f'{timer.stop(): .5f} sec')
print(f_norm[-1])




# X = torch.randn(2, 20, device=try_gpu())
# net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))
# net = net.to(device=try_gpu())
# y = net(X)
# print(y)
# print(net)
# print(net[0].weight.data.device)