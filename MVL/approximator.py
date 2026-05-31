import numpy as np


class Approximator:
    def __init__(self, lr:float=0.001):
        self.features = 2 * np.eye(7, dtype=np.float64)
        self.features = np.concatenate((self.features, np.ones((7, 1))), axis=1)
        self.features[-1, -1] += 1
        self.features[-1, -2] -= 1
        self.lr = lr
        self.weights = np.ones((8,), dtype=np.float64)
        self.weights[-2] *= 10
    
    def update(self, gradient:np.ndarray):
        self.weights += self.lr * gradient
        return np.linalg.norm(np.dot(self.features, self.weights[:, np.newaxis])[-1, 0])
