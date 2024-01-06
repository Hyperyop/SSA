import unittest
import numpy as np
from SSA import SSA
import matplotlib.pyplot as plt

class TestSSA(unittest.TestCase):
    def setUp(self):
        self.ssah = SSA()
        self.ssav = SSA(type="VSSA")
        self.data = np.random.randn(5, 1000)  # 5 features, 100 samples
        self.data_sin = np.sin(np.vstack([np.linspace(0, 2* (2*i+1) * np.pi , 1000) for i in range(5)]))
        self.data = self.data/2 + self.data_sin
    def test_fit(self):
        self.ssah.fit(self.data)
        self.assertIsNotNone(self.ssah.U)
        self.assertIsNotNone(self.ssah.weights)

    def test_transform(self):
        self.ssah.fit(self.data)
        transformed_data = self.ssah.transform(self.data)
        self.assertEqual(transformed_data.shape, self.data.shape)

    def test_fit_transform(self):
        transformed_data = self.ssav.fit_transform(self.data)
        rmse = np.sqrt(np.mean((transformed_data - self.data)**2))
        self.assertEqual(transformed_data.shape, self.data.shape)
    def test_sin(self):
        self.ssav.fit(self.data_sin)
        transformed_data = self.ssav.fit_transform(self.data_sin)
        rmse = np.sqrt(np.mean((transformed_data - self.data_sin)**2))
        print(f" RMSE: {rmse}")
        self.assertEqual(transformed_data.shape, self.data_sin.shape)
    def test_ssav(self):
        self.ssav.fit(self.data_sin)
        transformed_data = self.ssav.fit_transform(self.data_sin)
        rmse = np.sqrt(np.mean((transformed_data - self.data_sin)**2))
        self.assertEqual(transformed_data.shape, self.data.shape)

    def test_assertions(self):
        with self.assertRaises(AssertionError):
            self.ssah.fit(np.random.rand(100, 5))  # More features than samples
        with self.assertRaises(AssertionError):
            self.ssah.transform(np.random.rand(100))  # 1D array


if __name__ == '__main__':
    unittest.main()