from math import prod
from random import gauss

from .tensor import Tensor
from operator import add
from functools import reduce
from array import array



def randn(*shape):
    return Tensor([gauss(0, 1) for _ in range(prod(shape))]).reshape(*shape)


# Часть 1:

def ones(*shape):
    return tensor([1 for _ in range(shape[0])])


def tensor(data):
    return Tensor(data)

def zeros(*shape):
    return tensor([0 for _ in range(shape[0])])


# Часть 2:

def eye(n):
    return Tensor(reduce(add, [[1 if i == j else 0 for i in range(n)] for j in range(n)])).reshape(n, n)

# def to_categorical(y, num_classes):
#     d = dict()
#     iter_e = iter(eye(num_classes))
#     arr = []
#     for num_class in y:
#         if num_class in d.keys():
#             row = d[num_class]
#         else:
#             row = next(iter_e)
#             d[num_class] = row
#         arr.append(row._data)
#     return Tensor(reduce(add, arr)).reshape(len(arr), num_classes)


def to_categorical(y, num_classes):
    e = eye(num_classes)
    arr = []
    for num_class in y:
        arr.append(e[int(num_class)]._data)
    return Tensor(reduce(add, arr)).reshape(len(arr), num_classes)
