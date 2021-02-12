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

    def __pow__(self, pow):
        return DualTensor(np.power(self.a, pow), pow * self.b * np.power(self.a, pow - 1))

    def dot_right(self, other: np.ndarray):
        """ Right is the position of the DualTensor """
        return DualTensor(other @ self.a, other @ self.b)

    def dot_left(self, other: np.ndarray):
        """ Left is the position of the DualTensor """
        return DualTensor(self.a @ other, self.b @ other)

    def exp(self):
        """ Element-wise exponential """
        return DualTensor(np.exp(self.a), np.exp(self.a) * self.b)

    def ln(self):
        """ Element-wise natural log """
        return DualTensor(np.log(self.a), 1 / self.a * self.b)

    def multiply(self, other: np.ndarray):
        return DualTensor(self.a * other, self.b * other)

    def sum(self):
        return DualTensor(np.array(np.sum(self.a)), np.array(np.sum(self.b)))

    def max(self):
        return DualTensor(np.max(self.a, axis=1)[:,None], np.zeros((1, self.b.shape[1])))

    def min(self):
        return DualTensor(np.array(np.min(self.a)), np.min(self.b))

    def mean_row(self):
        return DualTensor(np.mean(self.a, axis=1)[:, None], np.mean(self.b, axis=1)[:, None])

    def std_row(self):
        return DualTensor(np.std(self.a, axis=1)[:, None], np.std(self.b, axis=1)[:, None])

    def sum_columns(self):
        """ Sums across columns """
        return DualTensor(np.sum(self.a, axis=1)[:, None], np.sum(self.b, axis=1)[:, None])

    def mean(self):
        """ Returns mean across DualTensor """
        return DualTensor(np.array(np.mean(self.a)), np.array(np.mean(self.b)))

    def non_negative(self):
        """ Returns a new DualTensor where each element is max(element, 0) """
        a = np.maximum(np.zeros_like(self.a), self.a)
        b = self.b.copy()
        b[np.where(a == 0)] = 0
        return DualTensor(a, b)

    def __str__(self) -> str:
        return f"{self.a} + {self.b}e"
