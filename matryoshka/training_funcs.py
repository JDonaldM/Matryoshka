'''
File contains functions used when training the NNs
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.optimizers import Adam
import os
import pathlib


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
            X (array) : The training data. Must have shape (nsamps, nfeatures).
        '''

        # Check shape of X.
        if len(X.shape) != 2:
            raise ValueError("X does not have the correct shape. Must have shape (nsamps, nfeatures)")

        # Calculate min. value and largest diff. of all samples of X along the
        #  0th axis. Both min_val and diff can be vectors if required.
        self.min_val = np.min(X, axis=0)
        self.diff = np.max(X, axis=0) - np.min(X, axis=0)

    def transform(self, X):
        '''
        Transform the data.

        Args:
            X (array) : The data to be transformed.

        Returns:
            Array containing the transformed data.
        '''
        x = np.subtract(X, self.min_val)
        return np.true_divide(x, self.diff)

    def inverse_transform(self, X):
        '''
        Inverse transform the data.

        Args:
            X (array) : The data to be transformed.

        Returns:
            Array containing the inverse transformed data.
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
            X (array) : The training data. Must have shape (nsamps, nfeatures).
        '''
        # Check shape of X.
        if len(X.shape) != 2:
            raise ValueError("X does not have the correct shape. Must have shape (nsamps, nfeatures)")

        # Make sure there are no negative values or zeros.
        if np.any(X<=0.):
            raise ValueError("X contains negative values or zeros.")

        X = np.log(X)

        # Calculate min. value and largest diff. of all samples of X along the
        #  0th axis. Both min_val and diff can be vectors if required.
        self.min_val = np.min(X, axis=0)
        self.diff = np.max(X, axis=0) - np.min(X, axis=0)

    def transform(self, X):
        '''
        Transform the data.

        Args:
            X (array) : The data to be transformed.

        Returns:
            Array containing the transformed data.
        '''
        X = np.log(X)
        x = np.subtract(X, self.min_val)
        return np.true_divide(x, self.diff)

    def inverse_transform(self, X):
        '''
        Inverse transform the data.

        Args:
            X (array) : The data to be transformed.

        Returns:
            Array containing the inverse transformed data.
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
            X (array) : The training data. Must have shape (nsamps, nfeatures).
        '''

        # Check shape of X.
        if len(X.shape) != 2:
            raise ValueError("X does not have the correct shape. Must have shape (nsamps, nfeatures).")

        # Calculate the mean and strandard deviation of X along the 0th axis.
        #  Can be vectors if needed.
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)

    def transform(self, X):
        '''
        Transform the data.

        Args:
            X (array) : The data to be transformed.

        Returns:
            Array containing the transformed data.
        '''
        x = np.subtract(X, self.mean)
        return np.true_divide(x, self.scale)

    def inverse_transform(self, X):
        '''
        Inverse transform the data.

        Args:
            X (array) : The data to be transformed.

        Returns:
            Array containing the inverse transformed data.
        '''
        x = np.multiply(X, self.scale)
        return np.add(x, self.mean)

class Resampler:
    '''
    Class for re-sampling the parameter space covered by a suite of simulations.
    The new samples can then be used to generate training data for the base model
    componenet emulators.

    .. note::
        See the `Generating training samples for the base model componenets
        <../example_notebooks/resample_example.ipynb>`_ example.

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
        if (simulation_samples is None) and (use_latent_space is True):
            raise ValueError("Latent space cannot be used without simulation samples.")
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

    def new_samples(self, nsamps, LH=True, buffer=None):
        '''
        Generate new samples from the region covered by the simulations.

        Args:
            nsamps (int) : The number of new samples to generate.
            LH (bool) : If True will use latin-hypercube sampling. Default
             is True.

        Returns:
            Array containing the new samples. Has shape (nsamps, d).
        '''
        if buffer is not None:
            self.min = self.min*(1-buffer)
            self.max = self.max*(1+buffer)
            self.diff = self.max - self.min

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

def trainNN(trainX, trainY, validation_data, nodes, learning_rate, batch_size, epochs, 
            callbacks=None, DR=None, verbose=0):

    '''
    A high-level function for quickly training a simple NN based emulator. The user
    NN will be optimsed with an Adam optimser and mean squared error loss function.

    Args:
        trainX (array) : Array containing the parameters/features of the training set.
         Should have shape (n, d).
        trainY (aray) : Array containing the target function of the training set.
         Should have shape (n, k).
        validation_data (tuple) : Tuple of arrays (valX, valY). Where `valX` and `valY`
         are the equivalent of `trainX` and `trainY` for the validation data. Can be 
         None if there is not a validation set.
        nodes (array) : Array containing the number of nodes in each hidden layer. 
         Should have shape (N, ), with N being the desired number of hidden layers.
        learning_rate (float) : The learning rate to be used during training.
        batch_size (int) : The batch size to be used during training.
        epochs (int) : The number of epochs to train the NN.
        callbacks (list) : List of `tensorflow` callbacks e.g. EarlyStopping
        DR (float) : Float between 0 and 1 that defines the dropout rate. If None
         dropout will not be used.
        verbose (int) : Defines how much information `tensorflow` prints during training.
          0 = silent, 1 = progress bar, 2 = one line per epoch.

    Returns:
        Trained keras Sequential model.
    '''

    # Define the NN as a keras Sequential model    
    model = Sequential()

    # Add the input layer
    model.add(InputLayer(input_shape=(trainX.shape[1], )))

    # Add the user specified number of hidden layers.
    for layer in range(nodes.shape[0]):
        model.add(Dense(nodes[layer], activation='relu'))
        if DR is not None:
            model.add(Dropout(DR))

    # Add the output layer
    model.add(Dense(trainY.shape[1], activation='linear'))

    # Complile the model with the user specified learning rate.
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))

    # Train the model
    model.fit(trainX, trainY, validation_data=validation_data, epochs=epochs, 
              batch_size=batch_size, callbacks = callbacks, verbose=verbose)

    return model

def dataset(target, split, X_or_Y):
    '''
    Convenience function for loading datasets for the base model component emulators.

    Args:
        target (str) : The target function of interest.
        split (str) : Can be "train", "test", or "val" (when a validation set is available).
        X_or_Y (str) : Do you want the features ("X") or the function ("Y").

    Returns:
        Array containing the dataset.
    '''
    cache_path = os.fsdecode(pathlib.Path(os.path.dirname(__file__)
                                          ).parent.absolute())+"/matryoshka-data/"
    cache_path += "class_aemulus/"
    return np.load(cache_path+split+"/"+X_or_Y+"_"+target+".npy")

def train_test_indices(N, split=0.2):
    '''
    Return indicies that can be used to split a dataset into train and test sets.

    Args:
        N (int) : The size of the original dataset
        split (float) : The proportion of the data to be used for the test set.
         Should be a float between 0 and 1. Default is 0.2

    Returns:
        The train and test indicies arrays.
    '''

    all = np.arange(N)
    np.random.shuffle(all)

    # How many samples in the test set
    N_test = int(split*N)

    return all[:N_test], all[N_test:]