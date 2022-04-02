import torch
from d2l import torch as d2l
from IPython import display

batch_size = 256  # 每个批次的大小
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 返回迭代器

"""读取数据"""
# trans = [transforms.ToTensor()]
# train_iter = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=False)
# test_iter = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=False)
# batch_size = 256
# train_iter = data.DataLoader(train_iter, batch_size, shuffle=True, num_workers=d2l.get_dataloader_workers())
# test_iter = data.DataLoader(test_iter, batch_size, shuffle=False, num_workers=d2l.get_dataloader_workers())


num_inputs = 784  # 输入特征数
num_outputs = 10  # 输出
w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)  # 初始化权重
b = torch.zeros(num_outputs, requires_grad=True)  # 初始化偏置

# X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n  # 使列表中存储n个0.0

    def add(self, *args):  # args为一个元组
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def net(X):
    """返回预测的标签值，即y_hat"""
    return softmax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)  # 把输入X拉平为一个列向量


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])  # len(y_hat)返回其行数，y为真实标签，y_hat为数据样本的预测概率


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # argmax(axis=1)获取每行中最大元素的索引
    cmp = y_hat.type(y.dtype) == y  # bool型,预测正确存为true，否则False
    return float(cmp.type(y.dtype).sum())  # 返回预测正确的数量


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):  # isinstance()判断net是否为torch.nn.Module类型
        net.eval()  # 将模型设置为评估模式。和net.train()的区别在于，eval()不启用Batch Normalization和Dropout
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():  # no_grad()不计算梯度，也不会进行反向传播
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())  # numel()获取tensor中元素的个数
    return metric[0] / metric[1]

print(evaluate_accuracy(net, test_iter))
# def train_epoch_ch3(net, train_iter, loss, updater):  #@save
#     """训练模型一个迭代周期（定义见第3章）"""
#     # 将模型设置为训练模式
#     if isinstance(net, torch.nn.Module):
#         net.train()
#     # 训练损失总和、训练准确度总和、样本数
#     metric = Accumulator(3)
#     for X, y in train_iter:
#         # 计算梯度并更新参数
#         y_hat = net(X)
#         l = loss(y_hat, y)
#         if isinstance(updater, torch.optim.Optimizer):
#             # 使用PyTorch内置的优化器和损失函数
#             updater.zero_grad()
#             l.mean().backward()
#             updater.step()
#         else:
#             # 使用定制的优化器和损失函数
#             l.sum().backward()
#             updater(X.shape[0])
#         metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
#     # 返回训练损失和训练精度
#     return metric[0] / metric[2], metric[1] / metric[2]