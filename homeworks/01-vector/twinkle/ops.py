from math import prod
from random import gauss

from .tensor import Tensor


def randn(*shape):
    return Tensor([gauss(0, 1) for _ in range(prod(shape))])


# Часть 1:

def ones(*shape):
    return tensor([1 for _ in range(shape[0])])


def tensor(data):
    return Tensor(data)


def zeros(*shape):
    return tensor([0 for _ in range(shape[0])])

