import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))  # 等价 y.sum().backward()
d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
# plt.show()

M = torch.normal(0, 1, size=(4, 4))
for i in range(100):
    M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))  # 二维tensor相乘
print(f"一百个矩阵相乘后:{M}")