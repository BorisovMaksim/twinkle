from array import array
from collections.abc import Iterable
import numbers

from .linalg import vector_norm


class Tensor(Iterable):
    def __init__(self, initializer):
        if isinstance(initializer, numbers.Number):
            self._data = initializer
            self._shape = ()
        elif isinstance(initializer, Iterable):
            for x in initializer:
                assert isinstance(x, numbers.Number), f"invalid data type '{type(x)}'"
            self._data = array('f', initializer)
            self._shape = (len(self._data),)
        else:
            raise TypeError(f"invalid data type '{type(initializer)}'")

    def __repr__(self):
        if len(self._shape) == 0:
            return f'Tensor({self._data:.4f})'
        return f"Tensor([{', '.join([f'{x:.4f}' for x in self._data])}])"

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
            return Tensor([fn(x, other) for x in self._data])
        if isinstance(other, Tensor):
            if self._shape == () and other._shape == ():
                return Tensor(fn(self._data, other._data))
            elif self._shape == ():
                return Tensor([fn(self._data, x) for x in other._data])
            elif other._shape == ():
                return Tensor([fn(x, other._data) for x in self._data])
            else:
                self._assert_same_shape(other)
                return Tensor([fn(self._data[i], other._data[i]) for i in range(len(self))])
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
        for x in self._data:
            yield x


    def __getitem__(self, key):
        if isinstance(key, numbers.Integral):
            return self._data[key]
        raise TypeError(f'only integers and 1-d Tensors are valid indices (got {type(key)})')
