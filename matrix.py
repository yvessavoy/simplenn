import random

from nn import NeuralNetwork


class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = []
        for row in range(rows):
            self.data.append([])
            for col in range(cols):
                self.data[row].append(0)

    def copy(self):
        m = Matrix(self.rows, self.cols)
        m.data = self.data.copy()
        return m

    @staticmethod
    def from_array(arr):
        return Matrix(len(arr), 1).map(lambda e, i, _: arr[i])

    @staticmethod
    def subtract(a, b):
        if a.rows != b.rows or a.cols != b.cols:
            print('Subtract error')
            return

        return Matrix(a.rows, a.cols).map(lambda _, i, j: a.data[i][j] - b.data[i][j])

    def to_array(self):
        arr = []
        for i in range(self.rows):
            for j in range(self.cols):
                arr.append(self.data[i][j])

        return arr

    def randomize(self):
        return self.map(lambda e, _, __: random.random() * 2 - 1)

    def add(self, n):
        if isinstance(n, Matrix):
            if self.rows != n.rows or self.cols != n.cols:
                print('Add error')
                return
            return self.map(lambda e, i, j: e + n.data[i][j])
        else:
            return self.map(lambda e, _, __: e + n)

    @staticmethod
    def transpose(matrix):
        return Matrix(matrix.cols, matrix.rows).map(lambda _, i, j: matrix.data[j][i])

    @staticmethod
    def static_multiply(a, b):
        if a.cols != b.rows:
            print('Multiply error')
            return
        return Matrix(a.rows, b.cols).map(lambda e, i, j: sum([a.data[i][k] * b.data[k][j] for k in range(a.cols)]))

    def multiply(self, n):
        if isinstance(n, Matrix):
            if self.rows != n.rows or self.cols != n.cols:
                print('Multiply error')
                return
            return self.map(lambda e, i, j: e * n.data[i][j])
        else:
            return self.map(lambda e, _, __: e * n)

    def map(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.data[i][j]
                self.data[i][j] = func(val, i, j)

        return self

    @staticmethod
    def static_map(matrix, func):
        return Matrix(matrix.rows, matrix.cols).map(lambda e, i, j: func(matrix.data[i][j], i, j))

