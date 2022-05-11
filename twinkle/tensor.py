from array import array
from collections.abc import Iterable
import numbers
from math import prod

from .linalg import vector_norm


class Tensor(Iterable):
    def __init__(self, initializer):
        if isinstance(initializer, numbers.Number):
            self._data = initializer
            self._shape = ()
        elif isinstance(initializer, Tensor):
            self._data = initializer._data
            self._shape = initializer._shape
        elif isinstance(initializer, Iterable):
            for x in initializer:
                assert isinstance(x, numbers.Number), f"invalid data type '{type(x)}'"
            self._data = array('f', initializer)
            self._shape = (len(self._data),)
        else:
            raise TypeError(f"invalid data type '{type(initializer)}'")

    def __repr__(self):
        if len(self._shape) == 0:
            return f"Tensor({self._data:.4f})"
        elif len(self._shape) == 1:
            return f"Tensor([{', '.join([f'{x:.4f}' for x in self._data])}])"
        elif len(self._shape) == 2:
            rows = [', '.join(f'{x:.4f}' for x in self._data[i * self._shape[1]: (i + 1) * self._shape[1]])\
                    for i in range(self._shape[0])]
            rows = ',\n        '.join(rows)
            return f"Tensor([{rows}]).reshape{self._shape}"
        return f"Tensor([{', '.join([f'{x:.4f}' for x in self._data])}]).reshape{self._shape}"

    def _assert_same_shape(self, other):
        assert isinstance(other, Tensor), f"argument 'other' must be Tensor, not {type(other)}"
        assert self._shape == other._shape, \
            f'The shape of tensor a ({self._shape}) must match the shape of tensor b ({other._shape})'

    def allclose(self, other, rtol=1e-05, atol=1e-08):
        self._assert_same_shape(other)
        return vector_norm(self - other) <= atol + vector_norm(rtol * other)

    # Часть 1:

    @property
    def shape(self):
        return self._shape

    def _binary_op(self, other, fn):
        if isinstance(other, numbers.Number):
            tensor = Tensor([fn(x, other) for x in self._data])
            tensor._shape = self.shape
            return tensor
        if isinstance(other, Tensor):
            if self._shape == () and other._shape == ():
                tensor = Tensor(fn(self._data, other._data))
                tensor._shape = self.shape
                return tensor
            elif self._shape == ():
                tensor = Tensor([fn(self._data, x) for x in other._data])
                tensor._shape = other._shape
                return tensor
            elif other._shape == ():
                tensor = Tensor([fn(x, other._data) for x in self._data])
                tensor._shape = self._shape
                return tensor
            else:
                self._assert_same_shape(other)
                tensor = Tensor([fn(self._data[i], other._data[i]) for i in range(prod(self.shape))])
                tensor._shape = self.shape
                return tensor
        raise TypeError(f"unsupported operand type(s) for +: 'Tensor' and '{type(other)}'")

    def add(self, other):
        return self._binary_op(other, lambda x, y: x + y)

    def mul(self, other):
        return self._binary_op(other, lambda x, y: x * y)

    def sub(self, other):
        return self._binary_op(other, lambda x, y: x - y)

    def lt(self, other):
        return self._binary_op(other, lambda x, y: int(x < y))

    def gt(self, other):
        return self._binary_op(other, lambda x, y: int(x > y))

    def neg(self):
        return self.mul(-1)

    def dot(self, other):
        self._assert_same_shape(other)
        assert len(self._shape) == 1, '1D tensors expected'
        return sum(self.mul(other))

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return self.sub(other).neg()

    def __gt__(self, other):
        return self.gt(other)

    def __lt__(self, other):
        return self.lt(other)

    def __neg__(self):
        return self.neg()

    def __len__(self):
        return self._shape[0]

    def __eq__(self, other):
        return self.allclose(other)

    def __iter__(self):
        assert len(self._shape) > 0, 'iteration over a 0-d tensor'
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, key):
        if isinstance(key, numbers.Integral):
            if len(self._shape) == 1:
                return self._data[key]
            sb_shape = self.shape[1:]
            step = prod(sb_shape)
            t = Tensor(self._data[key * step:(key + 1) * step]).reshape(*sb_shape)
            return t
        elif isinstance(key, Tensor) and len(key.shape) == 1:
            sb_shape = self.shape[1:]
            step = prod(sb_shape)
            t = Tensor(self._data[key._data[0] * step:(key._data[0] + 1) * step]).reshape(*sb_shape)
            return t
        raise TypeError(f'only integers and 1-d Tensors are valid indices (got {type(key)})')

    # Часть 2:

    def reshape(self, *shape):
        assert prod(shape) == len(self._data), \
            f"shape '[{shape}]' is invalid for input of size {len(self)}"
        tensor = Tensor(self._data)
        tensor._shape = shape
        return tensor

    def flatten(self):
        tensor = Tensor(self._data)
        tensor._shape = (prod(tensor._shape),)
        return tensor

    def argmax(self):
        mx = 0
        for i in range(len(self)):
            if self._data[i] > self._data[mx]:
                mx = i
        return mx


    def get_element(self, i, j):
        assert len(self._shape) == 2, 'self must be a matrix'
        return self._data[i * self.shape[1] + j]

    def mm(self, other):
        assert isinstance(other, Tensor), f"argument 'other' must be Tensor, not {type(other)}"
        assert len(self._shape) == 2, 'self must be a matrix'
        assert len(other._shape) == 2, 'other must be a matrix'
        assert self._shape[1] == other._shape[0], 'self and other shapes cannot be multiplied'
        tensor = Tensor([0 for _ in range(self.shape[0] * other.shape[1])]).reshape(self.shape[0], other.shape[1])
        # tensor._shape = (self.shape[0], other.shape[1])
        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                tensor._data[i * tensor.shape[1] + j] = \
                    sum([self.get_element(i, k) * other.get_element(k, j)
                         for k in range(self.shape[1])])
        return tensor

    def __matmul__(self, other):
        return self.mm(other)

    @property
    def T(self):
        t = Tensor(self._data).reshape(self.shape[1], self.shape[0])
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
              t._data[i * t.shape[1] + j] = self[j][i]
        return t
