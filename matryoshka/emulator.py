'''
Flie contains Classes for the idividual component emulators. In addition to a
 Class that combines all the component preictions to predict the galaxy power
 spectrum.
'''
from tensorflow.keras.models import load_model
import numpy as np
from .training_funcs import UniformScaler, LogScaler
#from halomod.concentration import Duffy08
#from hmf.halos.mass_definitions import SOMean
from .halo_model_funcs import Duffy08cmz
from . import halo_model_funcs
from . import eft_funcs
from scipy.interpolate import interp1d
import os
import pathlib

# Path to directory containing the NN weights as well as scalers needed produce
#  predictions with the NNs.
cache_path = os.fsdecode(pathlib.Path(os.path.dirname(__file__)
                                      ).parent.absolute())+"/matryoshka-data/"

# Define list of redshifts where there are trained NNs
matter_boost_zlist  = ['0', '0.5', '1']
galaxy_boost_zlist  = ['0.57']

# Define lists of relevant parameters for T(k) for each of the emulator versions.
relevant_transfer = {'class_aemulus':[0, 1, 3, 5, 6], 
                     'QUIP':[0, 1, 2]}

# Define some dictionaries that map which index of X_COSMO matches which parameter
# for the different emulator versions.
parameter_ids = {'class_aemulus':{'Om':0,'Ob':1,'sigma8':2,'h':3,'ns':4,'Neff':5,'w0':6},
                 'QUIP':{'Om':0,'Ob':1,'h':2,'ns':3,'sigma8':4}}

# Default k values where PyBird makes predictions. Needed by the EFT emulators.
kbird = np.array([0.001, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03,
                  0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085,
                  0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14,
                  0.145, 0.15, 0.155, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24,
                  0.25, 0.26, 0.27, 0.28, 0.29, 0.3])

class Transfer:
    '''
    Class for the transfer function componenet emulator.

    On initalisation the weights for the NN ensmble will be loaded,
    along with the scalers required to make predictions with the NNs.

    Args:
        version (str) : String to specify what version of the emulator to
         load. Default is 'class_aemulus'.

    .. note::
        See the `Basic emulator usage <../example_notebooks/transfer_basic.ipynb>`_
        example.
    '''

    def __init__(self, version='class_aemulus'):

        self.kbins = np.logspace(-4, 1, 300)
        '''The k-bins at which predictions will be made.'''

        self.relevant_params = relevant_transfer[version]

        models_path = cache_path+version+"/"+"models/transfer/"

        # Load the ensemble of NNs that makes up the T(k) emulator.
        models = list()
        for member in os.listdir(models_path):
            model = load_model(models_path+member,
                                            compile=False)
            models.append(model)

        self.models = models
        '''A list containing the individual ensemble members.'''

        scalers_path = cache_path+version+"/"+"scalers/transfer/"

        xscaler = UniformScaler()
        yscaler = LogScaler()

        # Load the variables that define the scalers.
        xmin_diff = np.load(scalers_path+"xscaler_min_diff.npy")
        ymin_diff = np.load(scalers_path+"yscaler_min_diff.npy")

        xscaler.min_val = xmin_diff[0, :]
        xscaler.diff = xmin_diff[1, :]

        yscaler.min_val = ymin_diff[0, :]
        yscaler.diff = ymin_diff[1, :]

        self.scalers = (xscaler, yscaler)

    def emu_predict(self, X, mean_or_full='mean'):
        '''
        Make predictions with the component emulator.

        Args:
            X (array) : Array containing the relevant input parameters. If making
             a single prediction should have shape (d,), if a batch prediction
             should have the shape (N,d).
            mean_or_full (str) : Can be either 'mean' or 'full'. Determines if the
             ensemble mean prediction should be returned, or the predictions
             from each ensemble member (default is 'mean').

        Returns:
            Array containing the predictions from the component emulator. Array
            will have shape (m,n,k). If mean_or_full='mean' will have shape (n,k).
        '''

        # If making a prediction on single parameter set, input array needs to
        # be reshaped.
        X = np.atleast_2d(X)[:,self.relevant_params]

        X_prime = self.scalers[0].transform(X)

        if mean_or_full == "mean":

            preds = 0
            for i in range(len(self.models)):
                preds += self.scalers[1].inverse_transform(
                    self.models[i](X_prime))

            return preds/float(len(self.models))

        elif mean_or_full == "full":

            preds = np.zeros(
                (len(self.models), X_prime.shape[0], self.kbins.shape[0]))
            for i in range(len(self.models)):
                preds[i, :, :] = self.scalers[1].inverse_transform(
                    self.models[i](X_prime))

            return preds


class Sigma:
    '''
    Class for the mass variance componenet emulator.

    On initalisation the weights for the NN ensmble will be loaded,
    along with the scalers required to make predictions with the NNs.

    Args:
        version (str) : String to specify what version of the emulator to
         load. Default is 'class_aemulus'.
    '''

    def __init__(self, version='class_aemulus'):

        # Assume that all versions use the same mass bins.
        # TODO: Make this more general.
        self.mbins = np.load(cache_path+"AEMULUS-class_ms-test.npy")
        '''The m-bins at which predictions will be made.'''

        models_path = cache_path+version+"/"+"models/sigma/"

        # Load the ensemble of NNs that makes up the sigma(m) emulator.
        models = list()
        for member in os.listdir(models_path):
            model = load_model(models_path+member,
                                            compile=False)
            models.append(model)

        self.models = models
        '''A list containing the individual ensemble members.'''

        scalers_path = cache_path+version+"/"+"scalers/sigma/"

        xscaler = UniformScaler()
        yscaler = LogScaler()

        # Load the variables that define the scalers.
        xmin_diff = np.load(scalers_path+"xscaler_min_diff.npy")
        ymin_diff = np.load(scalers_path+"yscaler_min_diff.npy")

        xscaler.min_val = xmin_diff[0, :]
        xscaler.diff = xmin_diff[1, :]

        yscaler.min_val = ymin_diff[0, :]
        yscaler.diff = ymin_diff[1, :]

        self.scalers = (xscaler, yscaler)

    def emu_predict(self, X, mean_or_full='mean'):
        '''
        Make predictions with the component emulator.

        Args:
            X (array) : Array containing the relevant input parameters. If making
             a single prediction should have shape (d,), if a batch prediction
             should have the shape (N,d).
            mean_or_full : Can be either 'mean' or 'full'. Determines if the
             ensemble mean prediction should be returned, or the predictions
             from each ensemble member (default is 'mean').

        Returns:
            Array containing the predictions from the component emulator. Array
            will have shape (m,n,k). If mean_or_full='mean' will have shape (n,k).
        '''

        # If making a prediction on single parameter set, input array needs to
        # be reshaped.
        X = np.atleast_2d(X)

        X_prime = self.scalers[0].transform(X)

        if mean_or_full == "mean":

            preds = 0
            for i in range(len(self.models)):
                preds += self.scalers[1].inverse_transform(
                    self.models[i](X_prime))

            return preds/float(len(self.models))

        elif mean_or_full == "full":

            preds = np.zeros(
                (len(self.models), X_prime.shape[0], self.mbins.shape[0]))
            for i in range(len(self.models)):
                preds[i, :, :] = self.scalers[1].inverse_transform(
                    self.models[i](X_prime))

            return preds


class SigmaPrime:
    '''
    Class for the mass variance logarithmic derviative componenet emulator.

    On initalisation the weights for the NN ensmble will be loaded,
    along with the scalers required to make predictions with the NNs.

    Args:
        version (str) : String to specify what version of the emulator to
         load. Default is 'class_aemulus'.
    '''

    def __init__(self, version='class_aemulus'):

        # Assume that all versions use the same mass bins.
        # TODO: Make this more general.
        self.mbins = np.load(cache_path+"AEMULUS-class_ms-test.npy")
        '''The m-bins at which predictions will be made.'''

        models_path = cache_path+version+"/"+"models/dlns/"

        # Load the ensemble of NNs that makes up the dlns(m) emulator.
        models = list()
        for member in os.listdir(models_path):
            model = load_model(models_path+member,
                                            compile=False)
            models.append(model)

        self.models = models
        '''A list containing the individual ensemble members.'''

        scalers_path = cache_path+version+"/"+"scalers/dlns/"

        xscaler = UniformScaler()
        yscaler = UniformScaler()

        # Load the variables that define the scalers.
        xmin_diff = np.load(scalers_path+"xscaler_min_diff.npy")
        ymin_diff = np.load(scalers_path+"yscaler_min_diff.npy")

        xscaler.min_val = xmin_diff[0, :]
        xscaler.diff = xmin_diff[1, :]

        yscaler.min_val = ymin_diff[0, :]
        yscaler.diff = ymin_diff[1, :]

        self.scalers = (xscaler, yscaler)

    def emu_predict(self, X, mean_or_full='mean'):
        '''
        Make predictions with the component emulator.

        Args:
            X (array) : Array containing the relevant input parameters. If making
             a single prediction should have shape (d,), if a batch prediction
             should have the shape (N,d).
            mean_or_full : Can be either 'mean' or 'full'. Determines if the
             ensemble mean prediction should be returned, or the predictions
             from each ensemble member (default is 'mean').

        Returns:
            Array containing the predictions from the component emulator. Array
            will have shape (m,n,k). If mean_or_full='mean' will have shape (n,k).
        '''

        # If making a prediction on single parameter set, input array needs to
        # be reshaped.
        X = np.atleast_2d(X)

        X_prime = self.scalers[0].transform(X)

        if mean_or_full == "mean":

            preds = 0
            for i in range(len(self.models)):
                preds += self.scalers[1].inverse_transform(
                    self.models[i](X_prime))

            return preds/float(len(self.models))

        elif mean_or_full == "full":

            preds = np.zeros(
                (len(self.models), X_prime.shape[0], self.mbins.shape[0]))
            for i in range(len(self.models)):
                preds[i, :, :] = self.scalers[1].inverse_transform(
                    self.models[i](X_prime))

            return preds


class Growth:
    '''
    Class for the growth function componenet emulator.

    On initalisation the weights for the NN ensmble will be loaded,
    along with the scalers required to make predictions with the NNs.

    Args:
        version (str) : String to specify what version of the emulator to
         load. Default is 'class_aemulus'.
    '''

    def __init__(self, version='class_aemulus'):

        # Assume that all versions use the same redshift bins.
        # TODO: Make this more general.
        self.zbins = np.linspace(0, 2, 200)
        '''The z-bins at which predictions will be made.'''

        self.relevant_params = relevant_transfer[version]

        models_path = cache_path+version+"/"+"models/growth/"

        # Load the ensemble of NNs that makes up the D(z) emulator.
        models = list()
        for member in os.listdir(models_path):
            model = load_model(models_path+member,
                                            compile=False)

            models.append(model)

        self.models = models
        '''A list containing the individual ensemble members.'''

        scalers_path = cache_path+version+"/"+"scalers/growth/"

        xscaler = UniformScaler()
        yscaler = LogScaler()

        # Load the variables that define the scalers.
        xmin_diff = np.load(scalers_path+"xscaler_min_diff.npy")
        ymin_diff = np.load(scalers_path+"yscaler_min_diff.npy")

        xscaler.min_val = xmin_diff[0, :]
        xscaler.diff = xmin_diff[1, :]

        yscaler.min_val = ymin_diff[0, :]
        yscaler.diff = ymin_diff[1, :]

        self.scalers = (xscaler, yscaler)

    def emu_predict(self, X, mean_or_full='mean'):
        '''
        Make predictions with the component emulator.

        Args:
            X (array) : Array containing the relevant input parameters. If making
             a single prediction should have shape (d,), if a batch prediction
             should have the shape (N,d).
            mean_or_full : Can be either 'mean' or 'full'. Determines if the
             ensemble mean prediction should be returned, or the predictions
             from each ensemble member (default is 'mean').

        Returns:
            Array containing the predictions from the component emulator. Array
            will have shape (m,n,k). If mean_or_full='mean' will have shape (n,k).
        '''

        # If making a prediction on single parameter set, input array needs to
        # be reshaped.
        X = np.atleast_2d(X)[:,self.relevant_params]

        X_prime = self.scalers[0].transform(X)

        if mean_or_full == "mean":

            preds = 0
            for i in range(len(self.models)):
                pred = self.scalers[1].inverse_transform(
                    self.models[i](X_prime))
                pred[:, 0] = 1.
                preds += pred

            return preds/float(len(self.models))

        elif mean_or_full == "full":

            preds = np.zeros(
                (len(self.models), X_prime.shape[0], self.zbins.shape[0]))
            for i in range(len(self.models)):
                preds[i, :, :] = self.scalers[1].inverse_transform(
                    self.models[i](X_prime))
                preds[i, :, 0] = 1.

            return preds


class Boost:
    '''
    Class for the nonlinear boost componenet emulator.

    On initalisation the weights for the NN ensmble will be loaded,
    along with the scalers required to make predictions with the NNs.

    Args: 
        redshift_id (int) : Index in matter_boost_zlist or galaxy_boost_zlist
         that corespons to the desired redshift.
    '''

    def __init__(self, redshift_id):

        # The scales where the Boost component emulator produces predictions is
        #  dependent on the simulation suite used to generate the training data.
        #  Currently based on the Aemulus suite.
        # TODO: Make this more generic.
        Lbox = 1050
        Nmesh = 1024
        k_ny = np.pi * Nmesh / Lbox
        k_fund = 2*np.pi / Lbox
        ksim = np.arange(k_fund, 0.5*k_ny, 2*k_fund)
        ksim = (ksim[:-1]+ksim[1:])/2.

        self.kbins = ksim
        '''The k-bins at which predictions will be made.'''

        boost_path = cache_path+"class_aemulus/boost_kwanspace_z{a}/".format(a=galaxy_boost_zlist[redshift_id])

        # Load the ensemble of NNs that makes up the B(k) emulator.
        models = list()
        for member in os.listdir(boost_path+"model"):
            model = load_model(boost_path+"model/"+member,
                                            compile=False)

            models.append(model)

        self.models = models
        '''A list containing the individual ensemble members.'''

        xscaler = UniformScaler()
        yscaler = LogScaler()

        # Load the variables that define the scalers.
        xmin_diff = np.load(boost_path+"scalers/xscaler_min_diff.npy")
        ymin_diff = np.load(boost_path+"scalers/yscaler_min_diff.npy")

        xscaler.min_val = xmin_diff[0, :]
        xscaler.diff = xmin_diff[1, :]

        yscaler.min_val = ymin_diff[0, :]
        yscaler.diff = ymin_diff[1, :]

        self.scalers = (xscaler, yscaler)

    def emu_predict(self, X, mean_or_full='mean'):
        '''
        Make predictions with the component emulator.

        Args:
            X (array) : Array containing the relevant input parameters. If making
             a single prediction should have shape (d,), if a batch prediction
             should have the shape (N,d).
            mean_or_full : Can be either 'mean' or 'full'. Determines if the
             ensemble mean prediction should be returned, or the predictions
             from each ensemble member (default is 'mean').

        Returns:
            Array containing the predictions from the component emulator. Array
            will have shape (m,n,k). If mean_or_full='mean' will have shape (n,k).
        '''

        # If making a prediction on single parameter set, input array needs to
        # be reshaped.
        X = np.atleast_2d(X)

        X_prime = self.scalers[0].transform(X)

        if mean_or_full == "mean":

            preds = 0
            for i in range(len(self.models)):
                preds += self.scalers[1].inverse_transform(
                    self.models[i](X_prime))

            return preds/float(len(self.models))

        elif mean_or_full == "full":

            preds = np.zeros(
                (len(self.models), X_prime.shape[0], self.kbins.shape[0]))
            for i in range(len(self.models)):
                preds[i, :, :] = self.scalers[1].inverse_transform(
                    self.models[i](X_prime))

            return preds

class MatterBoost:
    '''
    Emulator for predicting the nonlinear boost for the matter power
     spectrum in real space. Trained with the QUIJOTE simulations.

    Args:
        redshift_id (int) : Index in ``matter_boost_zlist``
         that corespons to the desired redshift. 
    '''

    def __init__(self, redshift_id):
        # Currently only trained on Quijote sims so defining the
        # kbins based on that.
        # TODO: MAke more general.
        k, _ = np.loadtxt(cache_path+'QUIP/Pk_m_z=0.txt',
                          unpack=True)
        ks_good = k < 1.0

        self.kbins = k[ks_good]
        '''The k-bins at which predictions will be made.'''

        self.redshift = float(matter_boost_zlist[redshift_id])

        models_path = cache_path+"QUIP/"+"models/"

        # Load the ensemble of NNs that makes up the B(k) emulator.
        models = list()
        for member in os.listdir(models_path+"boost_z{a}".format(a=matter_boost_zlist[redshift_id])):
            model = load_model(models_path+"boost_z{a}/".format(a=matter_boost_zlist[redshift_id])+member,
                                            compile=False)

            models.append(model)

        self.models = models
        '''A list containing the individual ensemble members.'''

        scalers_path = cache_path+"QUIP/"+"scalers/"

        xscaler = UniformScaler()
        yscaler = LogScaler()

        # Load the variables that define the scalers.
        xmin_diff = np.load(scalers_path+"boost_z{a}/xscaler_min_diff.npy".format(a=matter_boost_zlist[redshift_id]))
        ymin_diff = np.load(scalers_path+"boost_z{a}/yscaler_min_diff.npy".format(a=matter_boost_zlist[redshift_id]))

        xscaler.min_val = xmin_diff[0, :]
        xscaler.diff = xmin_diff[1, :]

        yscaler.min_val = ymin_diff[0, :]
        yscaler.diff = ymin_diff[1, :]

        self.scalers = (xscaler, yscaler)

    def emu_predict(self, X, mean_or_full='mean'):
        '''
        Make predictions with the component emulator.

        Args:
            X (array) : Array containing the relevant input parameters. If making
             a single prediction should have shape (d,), if a batch prediction
             should have the shape (N,d).
            mean_or_full : Can be either 'mean' or 'full'. Determines if the
             ensemble mean prediction should be returned, or the predictions
             from each ensemble member (default is 'mean').

        Returns:
            Array containing the predictions from the component emulator. Array
            will have shape (m,n,k). If mean_or_full='mean' will have shape (n,k).
        '''

        # If making a prediction on single parameter set, input array needs to
        # be reshaped.
        X = np.atleast_2d(X)

        X_prime = self.scalers[0].transform(X)

        if mean_or_full == "mean":

            preds = 0
            for i in range(len(self.models)):
                preds += self.scalers[1].inverse_transform(
                    self.models[i](X_prime))

            return preds/float(len(self.models))

        elif mean_or_full == "full":

            preds = np.zeros(
                (len(self.models), X_prime.shape[0], self.kbins.shape[0]))
            for i in range(len(self.models)):
                preds[i, :, :] = self.scalers[1].inverse_transform(
                    self.models[i](X_prime))

            return preds

class P11l:
    '''
    Class for emulator that predicts the P11l contributions to the
    P_n matrix.
    '''

    def __init__(self, multipole, version='EFTv2', redshift=0.51,
                 path_to_model=None, yscaler=None):

        if version=='EFTv3':
            self.kbins = kbird
        elif version=='custom':
            self.kbins = np.load(f"{path_to_model}kbins.npy")
        else:
            self.kbins = kbird[:39]

        if version=='custom':
            models_path = f"{path_to_model}/models/P11{multipole}/"
            xscalers_path = f"{path_to_model}/scalers/"
            yscalers_path = f"{path_to_model}/scalers/P11{multipole}/"
        else:            
            models_path = f"{cache_path}{version}/z{redshift}/models/P11{multipole}/"
            xscalers_path = f"{cache_path}{version}/z{redshift}/scalers/"
            yscalers_path = f"{cache_path}{version}/z{redshift}/scalers/P11{multipole}/"

        # Unlike many of the other matryoshka componenet emulators
        # the EFT components consist of just one NN.
        model = load_model(models_path+"member_0", compile=False)

        self.model = model
        '''The NN that forms this component emulator'''

        self.nonzero_cols = np.load(yscalers_path+"nonzero_cols.npy")
        '''There can be zeros for all cosmologies at certain k-values.
            The emulator does not make predictions here so we need to
            know where to put zeros.'''

        xscaler = UniformScaler()
        if yscaler is None:
            yscaler = UniformScaler()
        else:
            yscaler = yscaler

        # Load the variables that define the scalers.
        xmin_diff = np.load(xscalers_path+"xscaler_min_diff.npy")
        ymin_diff = np.load(yscalers_path+"yscaler_min_diff.npy")

        xscaler.min_val = xmin_diff[0, :]
        xscaler.diff = xmin_diff[1, :]

        yscaler.min_val = ymin_diff[0, :]
        yscaler.diff = ymin_diff[1, :]

        self.scalers = (xscaler, yscaler)
        self.multipole = multipole

    def emu_predict(self, X):
        '''
        Make predictions with the component emulator.

        Args:
            X (array) : Array containing the relevant input parameters. If making
             a single prediction should have shape ``(d,)``, if a batch prediction
             should have the shape ``(nsamp,d)``.

        Returns:
            Array containing the predictions from the component emulator 
            will have shape ``(nsamp,nkern,k)``.
        '''

        # If making a prediction on single parameter set, input array needs to
        # be reshaped.
        X = np.atleast_2d(X)

        X_prime = self.scalers[0].transform(X)

        preds = self.scalers[1].inverse_transform(
            self.model(X_prime))
        if self.multipole==2:
            preds -= 100.
        preds_incl_zeros = np.zeros((X.shape[0], 3*len(self.kbins)))
        preds_incl_zeros[:,self.nonzero_cols] = preds
        return preds_incl_zeros.reshape(X.shape[0],3,self.kbins.shape[0])

class Ploopl:
    '''
    Class for emulator that predicts the Ploopl contributions to the
    P_n matrix.
    '''

    def __init__(self, multipole, version='EFTv2', redshift=0.51,
                 path_to_model=None, yscaler=None):

        if version=='EFTv3':
            self.kbins = kbird
        elif version=='custom':
            self.kbins = np.load(f"{path_to_model}kbins.npy")
        else:
            self.kbins = kbird[:39]

        if version=='custom':
            models_path = f"{path_to_model}/models/Ploop{multipole}/"
            xscalers_path = f"{path_to_model}/scalers/"
            yscalers_path = f"{path_to_model}/scalers/Ploop{multipole}/"
        else:            
            models_path = f"{cache_path}{version}/z{redshift}/models/Ploop{multipole}/"
            xscalers_path = f"{cache_path}{version}/z{redshift}/scalers/"
            yscalers_path = f"{cache_path}{version}/z{redshift}/scalers/Ploop{multipole}/"

        # Unlike many of the other matryoshka componenet emulators
        # the EFT components consist of just one NN.
        model = load_model(models_path+"member_0", compile=False)

        self.model = model
        '''The NN that forms this component emulator'''

        self.nonzero_cols = np.load(yscalers_path+"nonzero_cols.npy")
        '''There can be zeros for all cosmologies at certain k-values.
           The emulator does not make predictions here so we need to
           know where to put zeros.'''

        xscaler = UniformScaler()
        if yscaler is None:
            yscaler = UniformScaler()
        else:
            yscaler = yscaler

        # Load the variables that define the scalers.
        xmin_diff = np.load(xscalers_path+"xscaler_min_diff.npy")
        ymin_diff = np.load(yscalers_path+"yscaler_min_diff.npy")

        xscaler.min_val = xmin_diff[0, :]
        xscaler.diff = xmin_diff[1, :]

        yscaler.min_val = ymin_diff[0, :]
        yscaler.diff = ymin_diff[1, :]

        self.scalers = (xscaler, yscaler)

    def emu_predict(self, X):
        '''
        Make predictions with the component emulator.

        Args:
            X (array) : Array containing the relevant input parameters. If making
             a single prediction should have shape ``(d,)``, if a batch prediction
             should have the shape ``(nsamp,d)``.

        Returns:
            Array containing the predictions from the component emulator 
            will have shape ``(nsamp,nkern,k)``.
        '''

        # If making a prediction on single parameter set, input array needs to
        # be reshaped.
        X = np.atleast_2d(X)

        X_prime = self.scalers[0].transform(X)

        preds = self.scalers[1].inverse_transform(
            self.model(X_prime))
        preds -= 10000
        preds_incl_zeros = np.zeros((X.shape[0], 12*len(self.kbins)))
        preds_incl_zeros[:,self.nonzero_cols] = preds
        return preds_incl_zeros.reshape(X.shape[0],12,self.kbins.shape[0])*np.array([-1., -1., 1., -1., 1., -1., 1., 1., 1., -1., -1., -1.])[np.newaxis,:,np.newaxis]

class Pctl:
    '''
    Class for emulator that predicts the Pctl contributions to the
    P_n matrix.
    '''

    def __init__(self, multipole, version='EFTv2' , redshift=0.51,
                 path_to_model=None, yscaler=None):

        if version=='EFTv3':
            self.kbins = kbird
        elif version=='custom':
            self.kbins = np.load(f"{path_to_model}kbins.npy")
        else:
            self.kbins = kbird[:39]

        if version=='custom':
            models_path = f"{path_to_model}/models/Pct{multipole}/"
            xscalers_path = f"{path_to_model}/scalers/"
            yscalers_path = f"{path_to_model}/scalers/Pct{multipole}/"
        else:            
            models_path = f"{cache_path}{version}/z{redshift}/models/Pct{multipole}/"
            xscalers_path = f"{cache_path}{version}/z{redshift}/scalers/"
            yscalers_path = f"{cache_path}{version}/z{redshift}/scalers/Pct{multipole}/"

        # Unlike many of the other matryoshka componenet emulators
        # the EFT components consist of just one NN.
        model = load_model(models_path+"member_0", compile=False)

        self.model = model
        '''The NN that forms this component emulator'''

        self.nonzero_cols = np.load(yscalers_path+"nonzero_cols.npy")
        '''There can be zeros for all cosmologies at certain k-values.
            The emulator does not make predictions here so we need to
            know where to put zeros.'''

        xscaler = UniformScaler()
        if yscaler is None:
            yscaler = UniformScaler()
        else:
            yscaler = yscaler

        # Load the variables that define the scalers.
        xmin_diff = np.load(xscalers_path+"xscaler_min_diff.npy")
        ymin_diff = np.load(yscalers_path+"yscaler_min_diff.npy")

        xscaler.min_val = xmin_diff[0, :]
        xscaler.diff = xmin_diff[1, :]

        yscaler.min_val = ymin_diff[0, :]
        yscaler.diff = ymin_diff[1, :]

        self.scalers = (xscaler, yscaler)
        self.multipole = multipole

    def emu_predict(self, X):
        '''
        Make predictions with the component emulator.

        Args:
            X (array) : Array containing the relevant input parameters. If making
             a single prediction should have shape ``(d,)``, if a batch prediction
             should have the shape ``(nsamp,d)``.

        Returns:
            Array containing the predictions from the component emulator 
            will have shape ``(nsamp,nkern,k)``.
        '''

        # If making a prediction on single parameter set, input array needs to
        # be reshaped.
        X = np.atleast_2d(X)

        X_prime = self.scalers[0].transform(X)

        preds = self.scalers[1].inverse_transform(
            self.model(X_prime))
        if self.multipole==2:
            preds -= 100.
        preds_incl_zeros = np.zeros((X.shape[0], 6*len(self.kbins)))
        preds_incl_zeros[:,self.nonzero_cols] = preds
        return preds_incl_zeros.reshape(X.shape[0],6,self.kbins.shape[0])

class EFT:
    '''
    Emulator for predicting power spectrum multipoles that would
    be predicted using EFTofLSS.

    Args:
        multipole (int) : Desired multipole. Can either be 0 or 2.
        version (str): Version of ``EFTEMU``. Can be ``EFTv2``, ``EFT-optiresum``,
         ``EFT_lowAs``, or ``custom``. Using ``custom`` requires ``path_to_model``
         and ``kmax`` to also be passed. Default is ``EFTv2``.
        redshift (float) : Desired redshift. Can be 0.38, 0.51, or 0.61.
         Default is 0.51.
        path_to_model (str) : Path where model trained with the ``trainEFTEMUcomponenets``
         script is saved. Only requred if ``version='custom'``. Default is ``None``.

    .. note::
        See the `EFTEMU <../example_notebooks/EFTEMU_example.ipynb>`_
        example.
    '''

    def __init__(self, multipole, version='EFTv2', redshift=0.51,
                 path_to_model=None, yscaler_P11=None, yscaler_Ploop=None, yscaler_Pct=None):

        if version=='custom' and (path_to_model is None):
            raise TypeError("when using version='custom' path_to_model must be specified.")

        if version=='custom' and (not os.path.isfile(f"{path_to_model}kbins.npy")):
            raise TypeError("when using version='custom' the directory at path_to_model must contaain a .npy file with the kbins the model predicts the kernels.")        

        self.P11 = P11l(multipole, version=version, redshift=redshift,
                        path_to_model=path_to_model, yscaler=yscaler_P11)
        '''The ``P_11`` component emulator.'''

        self.Ploop = Ploopl(multipole, version=version, redshift=redshift,
                            path_to_model=path_to_model, yscaler=yscaler_Ploop)
        '''The ``P_loop`` component emulator.'''

        self.Pct = Pctl(multipole, version=version, redshift=redshift,
                        path_to_model=path_to_model, yscaler=yscaler_Pct)
        '''The ``P_ct`` component emulator.'''

        self.multipole = multipole
        self.redshift = redshift

        self.param_names = ["w_c", "w_b", "h", "As", "ns"]
        '''List of the input parameters.'''

    def emu_predict(self, X, bias, stochastic=None, km=None, 
                    ng=None, kvals=None):
        '''
        Make predictions with the emulator.

        Args:
            X (array) : Input cosmological parameters.
             Should have shape (n, 5).
            bias (array) : Input bias parameters and counterterms. Should
             have shape (n, 7)
            stochastic (array) : Input stochastic counterterms. Should have
             shape (n, 3). Default is ``None``, in which case no stochastic
             terms are used.
            km (float) : Controls the bias derivative expansion (see eq. 5
             in arXiv:1909.05271). Default in ``None``, in which case all 
             counterterm inputs are assumed to be a ratio with km i.e.
             ``c_i/km**2``.
            ng (float) : Mean galaxy number density. Default is ``None``.
             Only required if ``stochastic`` is not ``None``.
            kvals (array) : Array containing k-values at which to produce predictions.
             Needs to be within the k-range that the emulator has been trained to
             predict. Default is ``None``, in which case predicts will be made at the
             default k-values.
        '''
        P11_preds = self.P11.emu_predict(X)
        Ploop_preds = self.Ploop.emu_predict(X)
        Pct_preds = self.Pct.emu_predict(X)

        # If making a prediction on single parameter set, input array needs to
        # be reshaped.
        X = np.atleast_2d(X)
        bias = np.atleast_2d(bias)

        if stochastic is not None:
            stochastic = np.atleast_2d(stochastic)
            if km is not None:
                # Make copy of array otherwise elements will be changed in place.
                stochastic = np.copy(stochastic)
                stochastic[:,1:] = stochastic[:,1:]/km**2

        if km is not None:
            bias = np.copy(bias)
            bias[:,4:] = bias[:,4:]/km**2

        f = halo_model_funcs.fN_vec((X[:,0]+X[:,1])/X[:,2]**2, self.redshift)
        multipole_array = eft_funcs.multipole_vec([P11_preds, Ploop_preds, Pct_preds],
                                                   bias, f.reshape(-1,1))

        # Iterpolate prediction with input k/
        if kvals is not None:
            # Check that inperpolation is possible.
            if kvals.max()<=self.P11.kbins.max() and kvals.min()>=self.P11.kbins.min():
                multipole_array =  interp1d(self.P11.kbins, multipole_array, 
                                            axis=-1)(kvals)
            else:
                #If not possible raise an error.
                raise ValueError("kvals need to be covered by default eulator range.")
        else:
            # Set the kvals variable if one isn't passed.
            # Allows kvals to alway be used for the stochastic terms below.
            kvals = self.P11.kbins

        if stochastic is not None:
            
            if self.multipole==0:
                multipole_array += stochastic[:,0].reshape(-1,1)/ng
                multipole_array += (stochastic[:,1].reshape(-1,1)*kvals**2)/ng

            elif self.multipole==2:    
                multipole_array += (stochastic[:,2].reshape(-1,1)*kvals**2)/ng

        return multipole_array

class QUIP:
    '''
    Emulator for predicting the real space nonlinear matter power spectrum. Trained 
    with the QUIJOTE simulations.

    Args:
        redshift_id (int) : Index in ``matter_boost_zlist``
         that corespons to the desired redshift.

    .. note::
        See the `QUIP <../example_notebooks/QUIP.ipynb>`_ example.
    '''

    def __init__(self, redshift_id):

        self.Transfer = Transfer(version='QUIP')
        '''The transfer function component emulator.'''

        self.MatterBoost = MatterBoost(redshift_id=redshift_id)
        '''The nonlinear boost component emulator.'''

        self.param_names = ["O_m", "O_b", "h", "ns", "sig8"]
        '''List of the input parameters.'''

    def emu_predict(self, X, kvals=None, mean_or_full='mean'):
        '''
        Make predictions with the emulator.

        Args:
            X (array) : Array containing the relevant input parameters. If making
             a single prediction should have shape ``(d,)``, if a batch prediction
             should have the shape ``(N,d)``.
            kvals (array) : Array containing k-values at which to produce predictions.
             Needs to be within the k-range that the emulator has been trained to
             predict. Default is ``None``, in which case predicts will be made at the
             default k-values.
            mean_or_full : Can be either 'mean' or 'full'. Determines if the
             ensemble mean prediction should be returned, or the predictions
             from each ensemble member (default is 'mean').

        Returns:
            Array containing the predictions from the emulator. Array
            will have shape ``(m,n,k)``. If ``mean_or_full='mean'`` will have shape ``(n,k)``.
        '''

        # If making a prediction on single parameter set, input array needs to
        # be reshaped.
        X = np.atleast_2d(X)

        transfer_preds = self.Transfer.emu_predict(X, mean_or_full=mean_or_full)
        boost_preds = self.MatterBoost.emu_predict(X, mean_or_full=mean_or_full)

        linPk0 = halo_model_funcs.power0_v2(self.Transfer.kbins, transfer_preds,
                                            sigma8=X[:, parameter_ids['QUIP']['sigma8']],
                                            ns=X[:, parameter_ids['QUIP']['ns']])

        growths = halo_model_funcs.DgN_vec(X[:, parameter_ids['QUIP']['Om']], self.MatterBoost.redshift)
        growths /= halo_model_funcs.DgN_vec(X[:, parameter_ids['QUIP']['Om']], 0.)

        linPk = interp1d(self.Transfer.kbins, linPk0, kind='cubic')(self.MatterBoost.kbins)\
                      *(growths**2).reshape(-1,1)

        if kvals is not None:
            if kvals.max()<self.MatterBoost.kbins.max() and kvals.min()>self.MatterBoost.kbins.min():
                return interp1d(self.MatterBoost.kbins, linPk*boost_preds)(kvals)
            else:
                raise ValueError("kvals need to be covered by default eulator range.")
        else:
            return linPk*boost_preds
           

class HaloModel:
    '''
    Class for the emulated halo model.

    Upon initalisation each of the component emulators will be initalised.

    Args:
        k (array) : The k-bins over which predictions will be made. Cannot be
         outside the ranges used when training the component emulators.
        redshift_id (int) : Index in matter_boost_zlist or galaxy_boost_zlist
         that corespons to the desired redshift. Only needed if nonlinear is True.
         Default is None.
        redshift (float) : The redshift at which predictions should be made. Can
         only be used if nonlinear is False. If nonlinear is True this will be ignored.
        nonlinear (bool) : Determines if nonlinear predictions should be made.
         If False, the nonlinear boost componenet emulator will not be
         initalised.
        matter (bool) : If nonlinear=True setting matter=True will use emulated
         nonlinear matter power. If matter=False the nonlinear boost will be
         applied to the galaxy power spectrum.
        version (str) : Version of the emulators to be loaded.
        kspace_filt (bool) : If True reduces contribution from P2h on small scales.
         Inspired by halomod. See section 2.9.1 of arXiv:2009.14066.
    '''

    def __init__(self, k, redshift_id=None, redshift=None, nonlinear=True, matter=True, 
                 version='class_aemulus', kspace_filt=False):

        # Initalise the base model components.
        self.Transfer = Transfer(version=version)
        self.sigma = Sigma(version=version)
        self.dlns = SigmaPrime(version=version)

        # Load the growth function emulator for non LCDM models.
        if version=='class_aemulus':
            self.growth = Growth()

        # Only load the nonlinear boost component if nonlinear predictions are
        #  required.
        self.nonlinear = nonlinear
        if nonlinear and matter:
            self.boost = MatterBoost(redshift_id)
            self.redshift = float(matter_boost_zlist[redshift_id])

        elif nonlinear:
            self.boost = Boost(redshift_id)
            self.redshift = float(galaxy_boost_zlist[redshift_id])

        else:
            self.redshift = redshift

        # Make sure desired prediction range is covered by the emulators.
        if k.min() < self.Transfer.kbins.min() or k.max() > self.Transfer.kbins.max():
            print("Input k outside emulator coverage! (LINEAR)")
        if nonlinear and k.max() > self.boost.kbins.max():
            print("Input k outside emulator coverage! (NONLINEAR)")

        if kspace_filt:
            self.filter = halo_model_funcs.TopHatrep(None, None)

        self.k = k
        self.version = version
        self.matter = matter
        

        # Initalise halmod mass defenition and calculate the conentration mass
        #  realtion.
        #md_mean = SOMean(overdensity=200)
        #duffy = Duffy08(mdef=md_mean)
        #conc_duffy = duffy.cm(self.sigma.mbins, z=redshift)
        conc_duffy = Duffy08cmz(self.sigma.mbins, self.redshift)
        self.cm = conc_duffy

    def emu_predict(self, X_COSMO, X_HOD, kspace_filt=False, RT=3.0):
        '''
        Make predictions for the halo model power spectrum with the
        pre-initalised component emulators.

        Args:
            X_COSMO (array) : Input cosmological parameters.
            X_HOD (array) : Input HOD parameters.

        Returns:
            Array containing the predictions from the halo model power spectrum.
            Array will have shape (n,k). If making a prediction for a single set 
            of input parameters will have shape (1,k).
        '''

        # Input must be reshaped if producing sinlge prediction.
        X_COSMO = np.atleast_2d(X_COSMO)
        X_HOD = np.atleast_2d(X_HOD)

        # Produce predictions from each of the components.
        T_preds = self.Transfer.emu_predict(X_COSMO,
                                            mean_or_full="mean")
        sigma_preds = self.sigma.emu_predict(X_COSMO,
                                             mean_or_full="mean")
        dlns_preds = self.dlns.emu_predict(X_COSMO,
                                           mean_or_full="mean")
        if self.version=='class_aemulus':
            gf_preds = self.growth.emu_predict(X_COSMO,
                                               mean_or_full="mean")

        if self.nonlinear and self.matter:
            boost_preds = self.boost.emu_predict(X_COSMO,
                                                 mean_or_full="mean")
            # Force the nonlinear boost to unity outside the emulation range.
            boost_preds = interp1d(self.boost.kbins, boost_preds, bounds_error=False,
                                   fill_value=1.0)(self.k)

        elif self.nonlinear:
            boost_preds = self.boost.emu_predict(np.hstack([X_HOD, X_COSMO]),
                                                 mean_or_full="mean")
            # Force the nonlinear boost to unity outside the emulation range.
            boost_preds = interp1d(self.boost.kbins, boost_preds, bounds_error=False,
                                   fill_value=1.0)(self.k)

        # Calculate the linear matter power spectrum at z=0 from the transfer
        #  function prediction.
        p_ml = halo_model_funcs.power0_v2(self.Transfer.kbins, T_preds, sigma8=X_COSMO[:, parameter_ids[self.version]['sigma8']],
                                          ns=X_COSMO[:, parameter_ids[self.version]['ns']])
        # Interpolate the power spectrum to cover the desired k-range.
        p_ml = interp1d(self.Transfer.kbins, p_ml)(self.k)

        if self.nonlinear and self.matter:
            p_ml = p_ml*boost_preds

        if kspace_filt:
            # Inspired by halomod.
            p_ml = p_ml*self.filter.k_space(self.k*RT)

        if self.version=='class_aemulus':
            # Interpolate the predicted growth function to return D(z) at the
            #  desired redshift.
            D_z = interp1d(self.growth.zbins, gf_preds)(self.redshift)

        else:
            D_z = np.zeros((p_ml.shape[0],))
            for i in range(D_z.shape[0]):
                # Assumes Om is in the first column of X_COSMO
                D_z[i] = halo_model_funcs.DgN(X_COSMO[i,0],self.redshift)/halo_model_funcs.DgN(X_COSMO[i,0],0.)

        # Produce HM galaxy power spectrum predictions using the component
        #  predictions.
        # TODO: I haven't found a nice way of vectorising the halo profile
        #  calculation. This loop currently dominates the prediction time so
        #  should be the first step when working on further optimisation.
        hm_preds = np.zeros((X_HOD.shape[0], self.k.shape[0]))
        n_ts = np.zeros((X_HOD.shape[0]))
        for i in range(X_HOD.shape[0]):
            # Create mass mask.
            tm = self.sigma.mbins >= X_HOD[i, 0] - 5*X_HOD[i, 1]

            Nc = halo_model_funcs.cen_Z09(
                self.sigma.mbins[tm], X_HOD[i, 0], X_HOD[i, 1])
            Ns = halo_model_funcs.sat_Z09(
                self.sigma.mbins[tm], X_HOD[i, 2], X_HOD[i, 4], X_HOD[i, 3], X_HOD[i, 0])
            Ntot = Nc*(1+Ns)

            mean_dens = halo_model_funcs.mean_density0_v2(
                h=X_COSMO[i, 3], Om0=X_COSMO[i, 0])
            halo_bias = halo_model_funcs.TinkerBias(
                np.sqrt(sigma_preds[i, tm]**2*D_z[i]**2))
            hmf = halo_model_funcs.hmf(
                sigma_preds[i, tm], dlns_preds[i, tm], mean_dens, self.sigma.mbins[tm], D_z[i], self.redshift)

            u_m = halo_model_funcs.u(
                self.k, self.sigma.mbins[tm], self.cm[tm], mean_dens, 200)

            n_t = halo_model_funcs.ngal(self.sigma.mbins[tm].reshape(
                1, -1), hmf.reshape(1, -1), Ntot.reshape(1, -1))[0]
            n_ts[i] = n_t

            P1h_ss = halo_model_funcs.power_1h_ss(
                u_m, hmf, self.sigma.mbins[tm], Nc, Ns, n_t)
            P1h_cs = halo_model_funcs.power_1h_cs(
                u_m, hmf, self.sigma.mbins[tm], Nc, Ns, n_t)
            P2h = halo_model_funcs.power_2h(
                u_m, hmf, self.sigma.mbins[tm], Ntot, n_t, p_ml[i]*D_z[i]**2, halo_bias)
            if self.nonlinear and not self.matter:
                # If making nonlinear predictions, combine the base model
                #  prediction with the boost component prediction.
                hm_preds[i, :] = (P2h+P1h_cs+P1h_ss)*boost_preds[i]
            else:
                hm_preds[i, :] = P2h+P1h_cs+P1h_ss
        return hm_preds, n_ts
