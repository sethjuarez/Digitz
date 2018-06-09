import os
import math
import numpy as np
import tensorflow as tf

class Digits:
    def __init__(self, data_dir, batch_size):
        # load MNIST data (if not available)
        self._data = os.path.join(data_dir, 'mnist.npz')
        self._train, self._test = tf.keras.datasets.mnist.load_data(path=self._data)
        self._batch_size = batch_size
        self._train_count = self._train[0].shape[0]
        self._size = self._train[0].shape[1] * self._train[0].shape[2]
        self._total = math.ceil((1. * self._train_count) / self._batch_size)

        self._testX = self._test[0].reshape(self._test[0].shape[0], self._size) / 255.
        self._testY = np.eye(10)[self._test[1]]

        self._trainX = self._train[0].reshape(self._train_count, self._size) / 255.
        self._trainY = self._train[1]
        

    def __iter__(self):
        # shuffle arrays
        p = np.random.permutation(self._trainX.shape[0])
        self._trainX = self._trainX[p]
        self._trainY = self._trainY[p]

        # reset counter
        self._current = 0

        return self

    def __next__(self):
        if self._current > self._train_count:
            raise StopIteration

        x = self._trainX[self._current : self._current + self._batch_size,:]
        y = np.eye(10)[self._trainY[self._current : self._current + self._batch_size]]

        if x.shape[0] == 0:
            raise StopIteration

        self._current += self._batch_size
        
        return x, y

    @property
    def test(self):
        return self._testX, self._testY

    @property
    def total(self):
        return self._total


if __name__ == "__main__":
    p = os.path.abspath('..\\data')
    print(p)
    digits = Digits(p, 1587)
    for i, (x, y) in enumerate(digits):
        print(i, x.shape, y.shape)

    print(digits.total)
    x, y = digits.test
    print(x.shape, y.shape)
    for i in x:
        print(i)
