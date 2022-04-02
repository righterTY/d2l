import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

# sigmoid
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
# sigmoid的导数
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
plt.show()
# relu
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
# relu的导数
y.backward(torch.ones_like(x), retain_graph=True)  # ones_like(x)创建一个和x一样维度，元素全为1的张量
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
plt.show()