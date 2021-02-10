import numpy as np
import tensor

Tensor = tensor.Tensor


class DualTensor:
    def __init__(self, a: Tensor, b: Tensor):
        self.a: Tensor = a
        self.b: Tensor = b

    def dot_right(self, other: Tensor):
        return DualTensor(other @ self.a, other @ self.b)

    def dot_left(self, other: Tensor):
        return DualTensor(self.a @ other, self.b @ other)


class Backward:
    def __init__(self):
        pass

    def backward(self, grad: Tensor):
        pass


class LinearBackprop(Backward):
    def __init__(self, X: Tensor, W: Tensor, b: Tensor):
        super().__init__()
        self.X: Tensor = X
        self.W: Tensor = W
        self.b: Tensor = b

    def backward(self, grad: Tensor) -> Tensor:
        # linear: vector -> vector
        # received: dL/dz
        # passed on: dL/dz * dz/da
        # dL/dW = dL/dz * dz/dW
        # dL/db = dL/dz summed throughout the batch
        return Tensor(np.array([]))


class ReLUBackprop(Backward):
    pass


class SoftmaxBackprop(Backward):
    pass


class CrossEntropyLoss(Backward):
    def __init__(self):
        super().__init__()


