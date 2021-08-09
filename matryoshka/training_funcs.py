'''
File contains functions used when training the NNs
'''
import numpy as np


class UniformScaler:
    '''
    Class for a simple uniform scaler. Linearly transforms X such that all
     samples in X are in the range [0,1].
    '''
    min_val = 0
    diff = 1

    def fit(self, X):
        # Calculate min. value and largest diff. of all samples of X along the
        #  0th axis. Both min_val and diff can be vectors if required.
        self.min_val = np.min(X, axis=0)
        self.diff = np.max(X, axis=0) - np.min(X, axis=0)

    def transform(self, X):
        x = np.subtract(X, self.min_val)
        return np.true_divide(x, self.diff)

    def inverse_transform(self, X):
        x = np.multiply(X, self.diff)
        return np.add(x, self.min_val)


class LogScaler:
    '''
    Class for a log scaler. Linearly transforms logX such that all samples in
     logX are in the range [0,1].
    '''
    min_val = 0
    diff = 1

    def fit(self, X):
        # Take the logarithm of X. This assumes that the log has not already
        #  been taken.
        X = np.log(X)

        self.min_val = np.min(X, axis=0)
        self.diff = np.max(X, axis=0) - np.min(X, axis=0)

    def transform(self, X):
        X = np.log(X)
        x = np.subtract(X, self.min_val)
        return np.true_divide(x, self.diff)

    def inverse_transform(self, X):
        x = np.multiply(X, self.diff)
        return np.exp(np.add(x, self.min_val))


class StandardScaler:
    '''
    Replacement for sklearn StandardScaler(). Rescales X such that it has zero
     mean and unit variance.
    '''
    mean = 0
    scale = 1

    def fit(self, X):
        # Calculate the mean and strandard deviation of X along the 0th axis.
        #  Can be vectors if needed.
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)

    def transform(self, X):
        x = np.subtract(X, self.mean)
        return np.true_divide(x, self.scale)

    def inverse_transform(self, X):
        x = np.multiply(X, self.scale)
        return np.add(x, self.mean)
