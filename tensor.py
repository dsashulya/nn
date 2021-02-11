import numpy as np
from typing import Callable, Optional


class Tensor:
    def __init__(self, data: np.ndarray, backprop = None):
        self.data: np.ndarray = data
        self.grad: np.ndarray = np.zeros_like(data)
        self.backprop = backprop

    def zero_grad(self):
        self.grad = np.zeros_like(self.grad)

    def backward(self, grad):
        if self.backprop is not None:
            self.backprop.backward(grad)

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        return Tensor(self.data.T)

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __sub__(self, other):
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        return Tensor(self.data * other.data)

    def __matmul__(self, other):
        return Tensor(self.data @ other.data)

    def __str__(self) -> str:
        return str(self.data)
