import numpy as np
from typing import List
from tensor import Tensor


class Module:
    def __init__(self):
        self.params: List[Tensor] = []

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()



class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.W: Tensor = Tensor(np.random.rand(in_features, out_features))
        self.b: Tensor = Tensor(np.random.rand(1, out_features))

        self.params.extend([self.W, self.b])

    def forward(self, X: Tensor) -> Tensor:
        return X @ self.W + self.b


class ReLU(Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(self, X: Tensor) -> Tensor:
        return Tensor(np.maximum(np.zeros_like(X.data), X.data))


class Softmax(Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(self, X: Tensor) -> Tensor:
        def softmax(x: np.ndarray) -> np.ndarray:
            # to avoid overflow
            x = x - np.max(x)
            return np.exp(x) / np.sum(np.exp(x))

        return Tensor(np.apply_along_axis(softmax, 1, X.data)) if len(X.shape) > 1 else Tensor(softmax(X.data))


