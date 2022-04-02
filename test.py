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