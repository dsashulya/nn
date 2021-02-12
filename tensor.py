import numpy as np
from typing import Union, Tuple


class DualNumber:
    def __init__(self, a: Union[float, int], b: Union[float, int]):
        self.a = a
        self.b = b

    def __mul__(self, other):
        return DualNumber(self.a * other.a, self.a * other.b + self.b * other.a)


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

    def sum_columns(self):
        """ Sums across columns """
        return DualTensor(np.sum(self.a, axis=1)[:, None], np.sum(self.b, axis=1)[:, None])

    def sum_rows(self):
        return DualTensor(np.sum(self.a), np.sum(self.b))

    def product(self, a: np.ndarray, b: np.ndarray) -> Tuple[Union[float, int], Union[float, int]]:
        total = DualNumber(a[0], b[0])
        for i in range(1, len(a)):
            total *= DualNumber(a[i], b[i])
        return (total.a, total.b)

    def product_rows(self):
        """ Product across rows """
        rows, _ = self.a.shape
        a, b = np.zeros(rows), np.zeros(rows)
        for row in range(rows):
            a[row], b[row] = self.product(self.a[row], self.b[row])

        return DualTensor(a, b)

    def product_columns(self):
        out = DualNumber(self.a[0], self.b[0])
        for el in range(1, self.a.shape[0]):
            out *= DualNumber(self.a[el], self.b[el])
        return DualTensor(np.array(out.a), np.array(out.b))

    def mean(self):
        """ Returns mean across DualTensor """
        return DualTensor(np.array(np.mean(self.a)), np.array(np.mean(self.b)))

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
