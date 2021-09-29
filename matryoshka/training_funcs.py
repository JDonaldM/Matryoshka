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
        '''
        Fit the parameters of the transformer based on the training data.

        Args:
            X (array) : The training data.
        '''
        # Calculate min. value and largest diff. of all samples of X along the
        #  0th axis. Both min_val and diff can be vectors if required.
        self.min_val = np.min(X, axis=0)
        self.diff = np.max(X, axis=0) - np.min(X, axis=0)

    def transform(self, X):
        '''
        Transform the data.

        Args:
            X (array) : The data to be transformed.
        '''
        x = np.subtract(X, self.min_val)
        return np.true_divide(x, self.diff)

    def inverse_transform(self, X):
        '''
        Inverse transform the data.

        Args:
            X (array) : The data to be transformed.
        '''
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
        '''
        Fit the parameters of the transformer based on the training data.

        Args:
            X (array) : The training data.
        '''
        # Take the logarithm of X. This assumes that the log has not already
        #  been taken.
        X = np.log(X)

        self.min_val = np.min(X, axis=0)
        self.diff = np.max(X, axis=0) - np.min(X, axis=0)

    def transform(self, X):
        '''
        Transform the data.

        Args:
            X (array) : The data to be transformed.
        '''
        X = np.log(X)
        x = np.subtract(X, self.min_val)
        return np.true_divide(x, self.diff)

    def inverse_transform(self, X):
        '''
        Inverse transform the data.

        Args:
            X (array) : The data to be transformed.
        '''
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
        '''
        Fit the parameters of the transformer based on the training data.

        Args:
            X (array) : The training data.
        '''
        # Calculate the mean and strandard deviation of X along the 0th axis.
        #  Can be vectors if needed.
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)

    def transform(self, X):
        '''
        Transform the data.

        Args:
            X (array) : The data to be transformed.
        '''
        x = np.subtract(X, self.mean)
        return np.true_divide(x, self.scale)

    def inverse_transform(self, X):
        '''
        Inverse transform the data.

        Args:
            X (array) : The data to be transformed.
        '''
        x = np.multiply(X, self.scale)
        return np.add(x, self.mean)

class Resampler:
    '''
    Class for re-sampling the parameter space covered by a suite of simulations.
    The new samples can then be used to generate training data for the base model
    componenet emulators.

    Args:
        simulation_samples (array) : The samples in the parameter space from the
         simulation suite. Default is None.
        parameter_ranges (array) : Ranges that define the extent of the parameter
         space. Should have shape (n, 2), where the first column is the minimum
         value for the n parameters, and the second column is the maximum.
         Default is None.
        use_latent_space (bool): If True the origonal simulation samples will be
         transfromed into an uncorrelated latent space for re-sampling. Default
         is False.
    '''
    def __init__(self, simulation_samples=None, parameter_ranges=None,
                 use_latent_space=False):
        
        # Make sure the user has passed either simulation_samples or parameter_ranges.
        if (simulation_samples is None) and (parameter_ranges is None):
            raise ValueError("Please provide either simulation samples or parameter ranges.")
        elif (parameter_ranges is None) and (use_latent_space is False):
            self.min = np.min(simulation_samples, axis=0)
            self.max = np.max(simulation_samples, axis=0)
            self.diff = self.max - self.min
            self.use_latent_space = use_latent_space
        elif (parameter_ranges is None) and (use_latent_space is True):
            self.L = np.linalg.cholesky(np.cov(simulation_samples, rowvar=False))
            self.use_latent_space = use_latent_space
            self.mean = np.mean(simulation_samples, axis=0)
            latent_samples = np.matmul(np.linalg.inv(self.L), (simulation_samples-self.mean).T).T
            self.min = latent_samples.min(axis=0)
            self.max = latent_samples.max(axis=0)
            self.diff = self.max - self.min
        elif parameter_ranges is not None:
            self.min = parameter_ranges[:,0]
            self.max = parameter_ranges[:,1]
            self.diff = self.max - self.min
            self.use_latent_space = use_latent_space

    def new_samples(self, nsamps, LH=True):
        '''
        Generate new samples from the region covered by the simulations.

        Args:
            nsamps (int) : The number of new samples to generate.
            LH (bool) : If True will use latin-hypercube sampling. Default
             is True.

        Returns:
            Array containing the new samples. Has shape (nsamps, d).
        '''

        if (LH is False) and (self.use_latent_space is False):
            return np.random.uniform(self.min, self.max, size=(nsamps,self.min.shape[0]))
        
        # How many dimensions in the sample space.
        d = self.min.shape[0]

        # Define the bin edges.
        low_edge = np.arange(0, nsamps)/nsamps
        high_edge = np.arange(1, nsamps+1)/nsamps

        # Generate the samples.
        latent_samples = np.random.uniform(low_edge, high_edge, (d, nsamps)).T
        for i in range(1,d):
            np.random.shuffle(latent_samples[:, i:])

        samples = np.zeros_like(latent_samples)
        for i in range(d):
            samples[:,i] = (latent_samples[:,i]*self.diff[i])+(self.min[i])

        if self.use_latent_space is False:
            return samples
        else:
            return np.matmul(self.L, samples.T).T+self.mean