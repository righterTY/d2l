<<<<<<< HEAD
import torch
import math
import numpy as np
from torch import nn
from d2l import torch as d2l
import hashlib
import zipfile
import tarfile
import os
import cv2

# path = './DATA1/test.zip'
# data_dir, ext = os.path.splitext(path)
# fname = os.path.dirname(path)
# if ext == '.zip':
#     fp = zipfile.ZipFile(path, 'r')
# elif ext in ('.tar', '.gz'):
#     fp = tarfile.open(fname, 'r')
# else:
#     assert False, "格式错误"
# fp.extractall(fname)

# class A():
#     def __init__(self, init_age):
#         self.age = init_age
#         print(f'年纪是{init_age}')
#
#     def __call__(self, added_age):
#         res = self.forward(added_age)
#         return res
#
#     def forward(self, input_):
#         print('forward被调用了')
#         return self.age + input_
#
#
# a = A(10)
# add_age = a(2)
# print(add_age)


# class Person():
#     def __init__(self, age, sex):
#         self.age = age
#         self.sex = sex
#         print('Person')
#
#     def print_title(self):
#         if self.sex == 'male':
#             print('man')
#         if self.sex == 'female':
#             print('woman')
#
#
# class Child(Person):
#     def __init__(self, age, sex, mather, father):
#         super().__init__(age, sex)
#         self.mather = mather
#         self.father = father
#
#
# class Baby(Child):
#     pass
#
#
# may = Person('49', 'male')
# peter = Child('18, 'man', 'Amy', 'Jason')
# # mike = Baby('8', 'female', 'Amy1', 'jason1')
# print(peter.mather)
# # print(mike.mather)
# # print(isinstance(may, Person))
# # print(isinstance(may, Child))
# # print(isinstance(peter, Person))

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.accumulate(args.integers))
=======
from torch import nn
import torch
from d2l import torch as d2l

net1 = nn.Sequential(
    # 这里，我们使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))

print(net1)
>>>>>>> b1fb6a71ce92adacf0966e5cb0cd6478dee83be3
