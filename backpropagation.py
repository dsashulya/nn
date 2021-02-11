import numpy as np
import tensor

Tensor = tensor.Tensor


class DualTensor:
    def __init__(self, a: Tensor, b: Tensor = None, position: int = None):
        """ position == -1 means Imaginary part equals 0 """
        self.a: Tensor = a
        if b is not None:
            self.b: Tensor = b
        else:
            assert isinstance(position, int)
            self.b: Tensor = Tensor(np.zeros_like(a.data))
            if position != -1:
                self.b[:, position] = 1

    def __add__(self, other):
        return DualTensor(self.a + other.a, self.b + other.b)

    def __sub__(self, other):
        return DualTensor(self.a - other.a, self.b - other.b)

    def dot_right(self, other: Tensor):
        """ right is the position of the dual tensor """
        return DualTensor(other @ self.a, other @ self.b)

    def dot_left(self, other: Tensor):
        """ left is the position of the dual tensor """
        return DualTensor(self.a @ other, self.b @ other)

    def __str__(self) -> str:
        return f"{self.a} + {self.b}e"


class Backward:
    """ Base class for all backpropagation modules """

    def __init__(self):
        pass

    def backward(self, grad: Tensor):
        pass


class LinearBackprop(Backward):
    """ Class for Linear module backpropagation """

    def __init__(self, X: Tensor, W: Tensor, b: Tensor):
        super().__init__()
        self.X: Tensor = X
        self.W: Tensor = W
        self.b: Tensor = b

    def fX(self, X: DualTensor) -> DualTensor:
        """ does a regular linear forward pass with a DualTensor argument """
        return X.dot_left(self.W) + DualTensor(self.b, Tensor(np.zeros_like(self.b.data)))

    def fW(self, W: DualTensor) -> DualTensor:
        """ does a linear forward pass with W as an input parameter in the DualTensor form """
        return W.dot_right(self.X) + DualTensor(self.b, Tensor(np.zeros_like(self.b.data)))

    def fb(self, b: DualTensor) -> DualTensor:
        """ does a linear forward pass with b as an input parameter in the DualTensor form """
        return DualTensor(self.X @ self.W, position=-1) + b

    def dX(self, grad: Tensor) -> Tensor:
        """ Calculates dL/da through dual numbers """
        batch_size, features = self.X.shape
        output = None
        for feature in range(features):
            # creates a dual number in the required position
            # outputs the np.ndarray derivative w.r.t the coordinate in position 'feature'
            dx = self.fX(DualTensor(self.X, position=feature)).b.data
            # taking the mean across the batch
            dx = np.mean(dx, axis=0)
            if output is None:
                output = dx.copy()[:, None]
            else:
                # concatenating with other feature derivatives
                output = np.concatenate((output, dx[:, None]), axis=1)
        return grad @ Tensor(output)

    def backward(self, grad: Tensor) -> Tensor:
        # linear: vector -> vector
        # received: dL/dz
        # passed on: dL/dz * dz/da
        # dL/dW = dL/dz * dz/dW
        # dL/db = dL/dz summed throughout the batch

        return self.dX(grad)


class ReLUBackprop(Backward):
    pass


class SoftmaxBackprop(Backward):
    pass


class CrossEntropyLoss(Backward):
    def __init__(self):
        super().__init__()


