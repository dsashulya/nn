import numpy as np


class DualTensor:
    def __init__(self, a: np.ndarray, b: np.ndarray = None):
        self.a: np.ndarray = a
        self.b: np.ndarray = b if b is not None else np.zeros_like(a)

    def __add__(self, other):
        return DualTensor(self.a + other.a, self.b + other.b)

    def __sub__(self, other):
        return DualTensor(self.a - other.a, self.b - other.b)

    def __truediv__(self, other):
        return DualTensor(self.a / other.a, (self.b * other.a - self.a * other.b) / np.square(other.a))

    def dot_right(self, other: np.ndarray):
        """ Right is the position of the DualTensor """
        return DualTensor(other @ self.a, other @ self.b)

    def dot_left(self, other: np.ndarray):
        """ Left is the position of the DualTensor """
        return DualTensor(self.a @ other, self.b @ other)

    def non_negative(self):
        """ Returns a new DualTensor where each element is max(element, 0) """
        a = np.maximum(np.zeros_like(self.a), self.a)
        b = self.b.copy()
        b[np.where(a == 0)] = 0
        return DualTensor(a, b)

    def __str__(self) -> str:
        return f"{self.a} + {self.b}e"
