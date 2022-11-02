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
import matplotlib.pyplot as plt

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
# mike = Baby('8', 'female', 'Amy1', 'jason1')
# print(peter.mather)
# print(mike.mather)
# print(isinstance(may, Person))
# print(isinstance(may, Child))
# print(isinstance(peter, Person))

x = np.linspace(-5*np.pi, 5*np.pi, 1000)
y = x*np.cos(np.pi*x)
plt.figure(num=3, figsize=(8, 5))
plt.plot(x, y, color='red', linewidth=1.0, linestyle='--')

# 设置 x，y 轴的范围以及 label 标注
plt.xlim(-10, 10)
plt.ylim(-20, 20)
plt.xlabel('x')
plt.ylabel('y')

# 设置坐标轴刻度线
# Tick X 范围 (-1，2) Tick Label(-1，-0.25，0.5，1.25，2) 刻度数量 5 个
new_ticks=np.linspace(-5*np.pi, 5*np.pi, 10)
plt.xticks(new_ticks)

# Tick Y 范围(-2.2,-1,1,1.5,2.4) ，Tick Label (-2.2, -1, 1, 1.5, 2.4) 别名(下面的英文)
new_ticks=np.linspace(-20, 20, 20)
plt.yticks(new_ticks)
# plt.yticks([-2.2,-1,1,1.5,2.4],[r'$really\ bad$',r'$bad$',r'$normal$',r'$good$',r'$really\ good$'])

# 设置坐标轴 gca() 获取坐标轴信息
ax=plt.gca()

# 使用.spines设置边框：x轴；将右边颜色设置为 none。
# 使用.set_position设置边框位置：y=0的位置；（位置所有属性：outward，axes，data）
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# 移动坐标轴
# 将 bottom 即是 x 坐标轴设置到 y=0 的位置。
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))

# 将 left 即是 y 坐标轴设置到 x=0 的位置。
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

# 设置标签
ax.set_title('y = x^2',fontsize=14,color='r')

# 显示图像
plt.show()