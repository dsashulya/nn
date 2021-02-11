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

    def exp(self):
        """ Element-wise exponential """
        return DualTensor(np.exp(self.a), np.exp(self.a) * self.b)

    def sum_columns(self):
        """ Sums across columns """
        return DualTensor(np.sum(self.a, axis=1)[:, None], np.sum(self.b, axis=1)[:, None])

    def non_negative(self):
        """ Returns a new DualTensor where each element is max(element, 0) """
        a, b = np.zeros_like(self.a), np.zeros_like(self.b)
        rows, cols = self.a.shape

        for i in range(rows):
            for j in range(cols):
                if self.a[i, j] < 0 or self.a[i, j] == 0 and self.b[i, j] < 0:
                    continue
                a[i, j], b[i, j] = self.a[i, j], self.b[i, j]
        return DualTensor(a, b)

    def __str__(self) -> str:
        return f"{self.a} + {self.b}e"
