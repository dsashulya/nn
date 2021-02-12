import numpy as np
from tensor import DualTensor
from typing import Callable, Optional


class Module:
    """ Base class for nn modules """
    def __init__(self):
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    def step(self, alpha: float):
        pass

    @staticmethod
    def derivative(X: np.ndarray, func: Callable, grad: np.ndarray,
                   one_dim: Optional[bool] = False) -> np.ndarray:
        # find derivatives w.r.t each element of X
        # scalar multiply grad by each of the derivatives to get dL/dx_i
        # assemble the matrix of all partials dL/dx_i
        output = np.zeros_like(X)

        if not one_dim:
            rows, cols = X.shape
            for i in range(rows):
                for j in range(cols):
                    b = np.zeros_like(X)
                    b[i, j] = 1
                    dx = func(DualTensor(X, b)).b
                    output[i, j] = np.sum(grad * dx)
        else:
            for i, _ in enumerate(X):
                b = np.zeros_like(output)
                b[i] = 1
                db = func(DualTensor(X, b)).b
                output[i] = np.sum(grad * db)
        return output

    def backward(self, grad: np.ndarray):
        pass


class Linear(Module):
    """ Class for Linear module backpropagation """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.W: np.ndarray = np.random.rand(in_features, out_features)
        self.b: np.ndarray = np.random.rand(out_features)

        self.X = None
        self.derivative_W = None
        self.derivative_b = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X: np.ndarray = X.copy()
        return X @ self.W + self.b

    def fX(self, X: DualTensor) -> DualTensor:
        """ does a regular linear forward pass with a DualTensor argument """
        return X.dot_left(self.W) + DualTensor(self.b, np.zeros_like(self.b))

    def fW(self, W: DualTensor) -> DualTensor:
        """ does a linear forward pass with W as an input parameter in the DualTensor form """
        return W.dot_right(self.X) + DualTensor(self.b, np.zeros_like(self.b))

    def fb(self, b: DualTensor) -> DualTensor:
        """ does a linear forward pass with b as an input parameter in the DualTensor form """
        return DualTensor(self.X @ self.W) + b

    def dX(self, grad: np.ndarray) -> np.ndarray:
        """ Calculates dL/da (same as dL/dX) through dual numbers """
        return self.derivative(self.X, self.fX, grad)

    def dW(self, grad: np.ndarray) -> np.ndarray:
        """ Calculates dL/dW through dual numbers """
        return self.derivative(self.W, self.fW, grad)

    def db(self, grad: np.ndarray) -> np.ndarray:
        """ Calculates dL/db through dual numbers """
        return self.derivative(self.b, self.fb, grad, one_dim=True)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # received: dL/dz
        # passed on: dL/dz * dz/da
        # dL/dW = dL/dz * dz/dW
        # dL/db = dL/dz * dz/db

        self.derivative_W = self.dW(grad)
        self.derivative_b = self.db(grad)
        return self.dX(grad)


    def step(self, alpha):
        self.b = self.b - alpha * self.derivative_b
        self.W = self.W - alpha * self.derivative_W


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.X = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X: np.ndarray = X.copy()
        return np.maximum(np.zeros_like(X), X)

    def fX(self, X: DualTensor) -> DualTensor:
        """ Forward pass using DualTensors """
        return X.non_negative()

    def dX(self, grad: np.ndarray) -> np.ndarray:
        return self.derivative(self.X, self.fX, grad)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self.dX(grad)


class Softmax(Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        # to avoid overflow
        x_ = x - np.max(x)
        return np.exp(x_) / np.sum(np.exp(x_))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self._softmax, 1, X) if len(X.shape) > 1 else self._softmax(X)


class CrossEntropyLoss(Module):
    def __init__(self, output: np.ndarray, target: np.ndarray, eps: Optional[float] = 1e-15):
        super().__init__()
        self.output: np.ndarray = output
        self.target: np.ndarray = target
        self.eps: float = eps

    @property
    def loss(self) -> float:
        log = np.log(self.output + self.eps)
        return np.sum(-self.target * log)
