import unittest

from matrix import Matrix


class TestMatrix(unittest.TestCase):
    def test_add_scalar_to_matrix(self):
        m = Matrix(3, 3)
        m.data[0] = [1, 2, 3]
        m.data[1] = [4, 5, 6]
        m.data[2] = [7, 8, 9]
        m.add(1)
        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 3)
        self.assertEqual(m.data, [[2, 3, 4], [5, 6, 7], [8, 9, 10]])

    def test_add_matrix_to_matrix(self):
        m = Matrix(2, 2)
        m.data[0] = [1, 2]
        m.data[1] = [3, 4]

        n = Matrix(2, 2)
        n.data[0] = [10, 11]
        n.data[1] = [12, 13]

        m.add(n)

        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)
        self.assertEqual(m.data, [[11, 13], [15, 17]])

    def test_subtract_matrix_from_matrix(self):
        m = Matrix(2, 2)
        m.data[0] = [10, 11]
        m.data[1] = [12, 13]

        n = Matrix(2, 2)
        n.data[0] = [1, 2]
        n.data[1] = [3, 4]
        
        m_minus_n = Matrix.subtract(m, n)

        self.assertEqual(m_minus_n.rows, 2)
        self.assertEqual(m_minus_n.cols, 2)
        self.assertEqual(m_minus_n.data, [[9, 9], [9, 9]])

    def test_matrix_product(self):
        m = Matrix(2, 3)
        m.data[0] = [1, 2, 3]
        m.data[1] = [4, 5, 6]

        n = Matrix(3, 2)
        n.data[0] = [7, 8]
        n.data[1] = [9, 10]
        n.data[2] = [11, 12]

        mn = Matrix.static_multiply(m, n)

        self.assertEqual(mn.rows, 2)
        self.assertEqual(mn.cols, 2)
        self.assertEqual(mn.data, [[58, 64], [139, 154]])

    def test_hadamard_product(self):
        m = Matrix(3, 2)
        m.data[0] = [1, 2]
        m.data[1] = [3, 4]
        m.data[2] = [5, 6]

        n = Matrix(3, 2)
        n.data[0] = [7, 8]
        n.data[1] = [9, 10]
        n.data[2] = [11, 12]

        m.multiply(n)

        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 2)
        self.assertEqual(m.data, [[7, 16], [27, 40], [55, 72]])

    def test_scalar_product(self):
        m = Matrix(3, 2)
        m.data[0] = [1, 2]
        m.data[1] = [3, 4]
        m.data[2] = [5, 6]

        m.multiply(7)

        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 2)
        self.assertEqual(m.data, [[7, 14], [21, 28], [35, 42]])

    def test_transpose_matrix_1(self):
        m = Matrix(1, 1)
        m.data[0] = [1]
        mt = Matrix.transpose(m)

        self.assertEqual(mt.rows, 1)
        self.assertEqual(mt.cols, 1)
        self.assertEqual(mt.data, [[1]])

    def test_transpose_matrix_2(self):
        m = Matrix(2, 3)
        m.data[0] = [1, 2, 3]
        m.data[1] = [4, 5, 6]

        mt = Matrix.transpose(m)

        self.assertEqual(mt.rows, 3)
        self.assertEqual(mt.cols, 2)
        self.assertEqual(mt.data, [[1, 4], [2, 5], [3, 6]])

    def test_transpose_matrix_3(self):
        m = Matrix(3, 2)
        m.data[0] = [1, 2]
        m.data[1] = [3, 4]
        m.data[2] = [5, 6]

        mt = Matrix.transpose(m)

        self.assertEqual(mt.rows, 2)
        self.assertEqual(mt.cols, 3)
        self.assertEqual(mt.data, [[1, 3, 5], [2, 4, 6]])

    def test_transpose_matrix_4(self):
        m = Matrix(1, 5)
        m.data[0] = [1, 2, 3, 4, 5]

        mt = Matrix.transpose(m)

        self.assertEqual(mt.rows, 5)
        self.assertEqual(mt.cols, 1)
        self.assertEqual(mt.data, [[1], [2], [3], [4], [5]])

    def test_transpose_matrix_5(self):
        m = Matrix(5, 1)
        m.data[0] = [1]
        m.data[1] = [2]
        m.data[2] = [3]
        m.data[3] = [4]
        m.data[4] = [5]

        mt = Matrix.transpose(m)

        self.assertEqual(mt.rows, 1)
        self.assertEqual(mt.cols, 5)
        self.assertEqual(mt.data, [[1, 2, 3, 4, 5]])

    def test_static_map(self):
        m = Matrix(3, 3)
        m.data[0] = [1, 2, 3]
        m.data[1] = [4, 5, 6]
        m.data[2] = [7, 8, 9]

        mapped = Matrix.static_map(m, lambda elem, _, __: elem * 10)

        self.assertEqual(mapped.rows, 3)
        self.assertEqual(mapped.cols, 3)
        self.assertEqual(mapped.data, [[10, 20, 30], [40, 50, 60], [70, 80, 90]])

    def test_instance_map(self):
        m = Matrix(3, 3)
        m.data[0] = [1, 2, 3]
        m.data[1] = [4, 5, 6]
        m.data[2] = [7, 8, 9]

        m.map(lambda elem, _, __: elem * 10)

        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 3)
        self.assertEqual(m.data, [[10, 20, 30], [40, 50, 60], [70, 80, 90]])

    def test_matrix_from_array(self):
        array = [1, 2, 3]
        m = Matrix.from_array(array)

        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 1)
        self.assertEqual(m.data, [[1], [2], [3]])

    def test_matrix_to_array(self):
        m = Matrix(3, 3)
        m.data[0] = [1, 2, 3]
        m.data[1] = [4, 5, 6]
        m.data[2] = [7, 8, 9]
        
        array = m.to_array()

        self.assertEqual(array, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_chaining_matrix_methods(self):
        m = Matrix(3, 3)
        m.data[0] = [1, 2, 3]
        m.data[1] = [4, 5, 6]
        m.data[2] = [7, 8, 9]

        m.map(lambda e, _, __: e - 1).multiply(10).add(6)

        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 3)
        self.assertEqual(m.data, [[6, 16, 26], [36, 46, 56], [66, 76, 86]])

    def test_instance_map_with_row_and_col(self):
        m = Matrix(3, 3)
        m.data[0] = [1, 2, 3]
        m.data[1] = [4, 5, 6]
        m.data[2] = [7, 8, 9]

        m.map(lambda e, row, col: e * 100 + row * 10 + col)

        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 3)
        self.assertEqual(m.data, [[100, 201, 302], [410, 511, 612], [720, 821, 922]])

    def test_static_map_with_row_and_col(self):
        m = Matrix(3, 3)
        m.data[0] = [1, 2, 3]
        m.data[1] = [4, 5, 6]
        m.data[2] = [7, 8, 9]

        mapped = Matrix.static_map(m, lambda e, row, col: e * 100 + row * 10 + col)

        self.assertEqual(mapped.rows, 3)
        self.assertEqual(mapped.cols, 3)
        self.assertEqual(mapped.data, [[100, 201, 302], [410, 511, 612], [720, 821, 922]])

    def test_matrix_copy(self):
        m = Matrix(5, 5)
        m.randomize()

        n = m.copy()

        self.assertEqual(n.rows, m.rows)
        self.assertEqual(n.cols, m.cols)
        self.assertEqual(n.data, m.data)


if __name__ == '__main__':
    unittest.main()
