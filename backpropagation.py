import numpy as np
import tensor

Tensor = tensor.Tensor


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
        return Tensor(np.array([]))


class CrossEntropyLoss(Backward):
    def __init__(self):
        super().__init__()


