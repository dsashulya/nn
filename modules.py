import numpy as np
from typing import List
from tensor import DualTensor


class Module:
    def __init__(self):
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.zeros_like(X)

class Linear(Module):
    """ Class for Linear module backpropagation """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.W: np.ndarray = np.random.rand(in_features, out_features)
        self.b: np.ndarray = np.random.rand(1, out_features)

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
        # find derivatives w.r.t each element of X
        # scalar multiply grad by each of the derivatives to get dL/dx_i
        # assemble the matrix of all partials dL/dx_i
        rows, cols = self.X.shape
        output = np.zeros_like(self.X)
        for i in range(rows):
            for j in range(cols):
                # create DualTensor with a dual number in position [i,j]
                b = np.zeros_like(self.W)
                b[i, j] = 1
                dx = self.fX(DualTensor(self.X, b)).b
                output[i, j] = np.sum(grad * dx)
        return output

    def dW(self, grad: np.ndarray) -> np.ndarray:
        """ Calculates dL/dW through dual numbers """
        # find derivatives w.r.t each element of W
        # scalar multiply grad by each of the derivatives to get dL/dw_i
        # assemble the matrix of all partials dL/dw_i
        rows, cols = self.W.shape
        output = np.zeros_like(self.W)
        for i in range(rows):
            for j in range(cols):
                # create DualTensor with a dual number in position [i,j]
                b = np.zeros_like(self.W)
                b[i, j] = 1
                # outputs the np.ndarray derivative w.r.t the element in position [i,j]
                dw = self.fW(DualTensor(self.W, b)).b
                output[i, j] = np.sum(grad * dw)
        return output

    def db(self, grad: np.ndarray) -> np.ndarray:
        """ Calculates dL/db through dual numbers """
        output = np.zeros_like(self.b)
        for i, _ in enumerate(self.b):
            b = np.zeros_like(output)
            b[i] = 1
            db = self.fb(DualTensor(self.b, b)).b
            output[i] = np.sum(grad * db)
        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # received: dL/dz
        # passed on: dL/dz * dz/da
        # dL/dW = dL/dz * dz/dW
        # dL/db = dL/dz * dz/db
        self.dw = self.dW(grad)
        self.db = self.db(grad)
        return self.dX(grad)


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X: np.ndarray = X.copy()
        return np.maximum(np.zeros_like(X), X)

    def fX(self, X: DualTensor) -> DualTensor:
        """ Forward pass using DualTensors """
        return X.non_negative()

    def dX(self, grad: np.ndarray) -> np.ndarray:
        output = np.zeros_like(self.X)
        rows, cols = self.X.shape
        for i in range(rows):
            for j in range(cols):
                b = np.zeros_like(self.X)
                b[i, j] = 1
                dx = self.fX(DualTensor(self.X, b)).b
                output[i, j] = np.sum(grad * dx)
        return output

    def backward(self, grad: np.ndarray):
        return self.dX(grad)


class Softmax(Module):
    def __init__(self, X: np.ndarray):
        super().__init__()

    def _softmax(x: np.ndarray) -> np.ndarray:
        # to avoid overflow
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    def forward(self, X: np.ndarray):
        self.X = X.copy()
        return np.apply_along_axis(self._softmax, 1, X) if len(X.shape) > 1 else self._softmax(X)


X = np.array([[1, 1], [-2, 3]])
W = np.array([[1, 2], [3, 4]])
b = np.array([1, 1])
l = ReLU()
l.forward(X)
print(l.backward(np.array([[1, 1], [1, 1]])))