'''
Flie contains Classes for the idividula component emulators. In addition to a
 Class that combines all the component preictions to predict the galaxy power
 spectrum.
'''
from tensorflow import keras
import numpy as np
from .training_funcs import UniformScaler, LogScaler
from halomod.concentration import Duffy08
from hmf.halos.mass_definitions import SOMean
from . import halo_model_funcs
from scipy.interpolate import interp1d

# Path to directory containing the NN weights as well as scalers needed produce
#  predictions with the NNs.
cache_path = "/Users/jamie/Desktop/Matryoshka/matryoshka-data/"
models_path = cache_path+"class_aemulus/models-v3/"
scalers_path = cache_path+"class_aemulus/scalers-v3/"
boost_path = cache_path+"class_aemulus/boost_kwanspace_z0.57/"


class Transfer:
    '''
    Class for the transfer function componenet emulator.
    '''

    def __init__(self):
        '''
        On initalisation the weights for the NN ensmble will be loaded,
         along with the scalers required to make predictions with the NNs.
        '''
        self.kbins = np.logspace(-4, 1, 300)

        # Load the ensemble of NNs that makes up the T(k) emulator.
        models = list()
        for i in range(11):
            model = keras.models.load_model(models_path+"transfer/member_"+str(i),
                                            compile=False)
            models.append(model)
        self.models = models

        xscaler = UniformScaler()
        yscaler = LogScaler()

        # Load the variables that define the scalers.
        xmin_diff = np.load(scalers_path+"transfer/xscaler_min_diff.npy")
        ymin_diff = np.load(scalers_path+"transfer/yscaler_min_diff.npy")

        xscaler.min_val = xmin_diff[0, :]
        xscaler.diff = xmin_diff[1, :]

        yscaler.min_val = ymin_diff[0, :]
        yscaler.diff = ymin_diff[1, :]

        self.scalers = (xscaler, yscaler)

    def emu_predict(self, X, single_or_batch="batch", mean_or_full="full"):

        # If making a prediction on single parameter set, input array needs to
        # be reshaped.
        if single_or_batch == "single":

            X = X.reshape(1, -1)

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
    '''

    def __init__(self):
        '''
        On initalisation the weights for the NN ensmble will be loaded,
         along with the scalers required to make predictions with the NNs.
        '''
        self.mbins = np.load(cache_path+"AEMULUS-class_ms-test.npy")

        # Load the ensemble of NNs that makes up the sigma(m) emulator.
        models = list()
        for i in range(15):
            model = keras.models.load_model(models_path+"sigma/member_"+str(i),
                                            compile=False)
            models.append(model)
        self.models = models

        xscaler = UniformScaler()
        yscaler = LogScaler()

        # Load the variables that define the scalers.
        xmin_diff = np.load(scalers_path+"sigma/xscaler_min_diff.npy")
        ymin_diff = np.load(scalers_path+"sigma/yscaler_min_diff.npy")

        xscaler.min_val = xmin_diff[0, :]
        xscaler.diff = xmin_diff[1, :]

        yscaler.min_val = ymin_diff[0, :]
        yscaler.diff = ymin_diff[1, :]

        self.scalers = (xscaler, yscaler)

    def emu_predict(self, X, single_or_batch="batch", mean_or_full="full"):

        # If making a prediction on single parameter set, input array needs to
        # be reshaped.
        if single_or_batch == "single":

            X = X.reshape(1, -1)

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
    '''

    def __init__(self):
        '''
        On initalisation the weights for the NN ensmble will be loaded,
         along with the scalers required to make predictions with the NNs.
        '''
        self.mbins = np.load(cache_path+"AEMULUS-class_ms-test.npy")

        # Load the ensemble of NNs that makes up the dlns(m) emulator.
        models = list()
        for i in range(8):
            model = keras.models.load_model(models_path+"dlnsdlnm/member_"+str(i),
                                            compile=False)

            models.append(model)
        self.models = models

        xscaler = UniformScaler()
        yscaler = UniformScaler()

        # Load the variables that define the scalers.
        xmin_diff = np.load(scalers_path+"dlnsdlnm/xscaler_min_diff.npy")
        ymin_diff = np.load(scalers_path+"dlnsdlnm/yscaler_min_diff.npy")

        xscaler.min_val = xmin_diff[0, :]
        xscaler.diff = xmin_diff[1, :]

        yscaler.min_val = ymin_diff[0, :]
        yscaler.diff = ymin_diff[1, :]

        self.scalers = (xscaler, yscaler)

    def emu_predict(self, X, single_or_batch="batch", mean_or_full="full"):

        # If making a prediction on single parameter set, input array needs to
        # be reshaped.
        if single_or_batch == "single":

            X = X.reshape(1, -1)

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
    '''

    def __init__(self):
        '''
        On initalisation the weights for the NN ensmble will be loaded,
         along with the scalers required to make predictions with the NNs.
        '''
        self.zbins = np.linspace(0, 2, 200)

        # Load the ensemble of NNs that makes up the D(z) emulator.
        models = list()
        for i in range(19):
            model = keras.models.load_model(models_path+"growth/member_"+str(i),
                                            compile=False)

            models.append(model)
        self.models = models

        xscaler = UniformScaler()
        yscaler = LogScaler()

        # Load the variables that define the scalers.
        xmin_diff = np.load(scalers_path+"growth/xscaler_min_diff.npy")
        ymin_diff = np.load(scalers_path+"growth/yscaler_min_diff.npy")

        xscaler.min_val = xmin_diff[0, :]
        xscaler.diff = xmin_diff[1, :]

        yscaler.min_val = ymin_diff[0, :]
        yscaler.diff = ymin_diff[1, :]

        self.scalers = (xscaler, yscaler)

    def emu_predict(self, X, single_or_batch="batch", mean_or_full="full"):

        # If making a prediction on single parameter set, input array needs to
        # be reshaped.
        if single_or_batch == "single":

            X = X.reshape(1, -1)

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
    '''

    def __init__(self):
        '''
        On initalisation the weights for the NN ensmble will be loaded,
         along with the scalers required to make predictions with the NNs.
        '''
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

        # Load the ensemble of NNs that makes up the B(k) emulator.
        models = list()
        for i in range(10):
            model = keras.models.load_model(boost_path+"model/member_"+str(i),
                                            compile=False)

            models.append(model)
        self.models = models

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

    def emu_predict(self, X, single_or_batch="batch", mean_or_full="full"):

        # If making a prediction on single parameter set, input array needs to
        # be reshaped.
        if single_or_batch == "single":

            X = X.reshape(1, -1)

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


class HaloModel:
    '''
    Class for the emulated halo model. Can be used to predictions from the base
     model, or the full suite of suite of emulators.
    '''

    def __init__(self, k, redshift, nonlinear=True):

        # Initalise the base model components.
        self.Transfer = Transfer()
        self.sigma = Sigma()
        self.growth = Growth()
        self.dlns = SigmaPrime()

        # Only load the nonlinear boost component if nonlinear predictions are
        #  required.
        self.nonlinear = nonlinear
        if nonlinear:
            self.boost = Boost()

        # Make sure desired prediction range is covered by the emulators.
        if k.min() < self.Transfer.kbins.min() or k.max() > self.Transfer.kbins.max():
            print("Input k outside emulator coverage! (LINEAR)")
        if nonlinear and k.max() > self.boost.kbins.max():
            print("Input k outside emulator coverage! (NONLINEAR)")

        self.k = k
        self.redshift = redshift

        # Initalise halmod mass defenition and calculate the conentration mass
        #  realtion.
        md_mean = SOMean(overdensity=200)
        duffy = Duffy08(mdef=md_mean)
        conc_duffy = duffy.cm(self.sigma.mbins, z=redshift)
        self.cm = conc_duffy

    def emu_predict(self, X_COSMO, X_HOD, single_or_batch="batch"):

        # Input must be reshaped if producing sinlge prediction.
        if single_or_batch == "single":
            X_COSMO = X_COSMO.reshape(1, -1)
            X_HOD = X_HOD.reshape(1, -1)

        # Produce predictions from each of the components.
        T_preds = self.Transfer.emu_predict(X_COSMO[:, [0, 1, 3, 5, 6]], single_or_batch="batch",
                                            mean_or_full="mean")
        sigma_preds = self.sigma.emu_predict(X_COSMO, single_or_batch="batch",
                                             mean_or_full="mean")
        dlns_preds = self.dlns.emu_predict(X_COSMO, single_or_batch="batch",
                                           mean_or_full="mean")
        gf_preds = self.growth.emu_predict(X_COSMO[:, [0, 1, 3, 5, 6]], single_or_batch="batch",
                                           mean_or_full="mean")
        if self.nonlinear:
            boost_preds = self.boost.emu_predict(np.hstack([X_HOD, X_COSMO]), single_or_batch="batch",
                                                 mean_or_full="mean")
            # Force the nonlinear boost to unity outside the emulation range.
            boost_preds = interp1d(self.boost.kbins, boost_preds, bounds_error=False,
                                   fill_value=1.0)(self.k)

        # Calculate the linear matter power spectrum at z=0 from the transfer
        #  function prediction.
        p_ml = halo_model_funcs.power0_v2(
            self.Transfer.kbins, T_preds, sigma8=X_COSMO[:, 2], ns=X_COSMO[:, 4])
        # Interpolate the power spectrum to cover the desired k-range.
        p_ml = interp1d(self.Transfer.kbins, p_ml)(self.k)

        # Interpolate the predicted growth function to return D(z) at the
        #  desired redshift.
        D_z = interp1d(self.growth.zbins, gf_preds)(self.redshift)

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
            if self.nonlinear:
                # If making nonlinear predictions, combine the base model
                #  prediction with the boost component prediction.
                hm_preds[i, :] = (P2h+P1h_cs+P1h_ss)*boost_preds[i]
            else:
                hm_preds[i, :] = P2h+P1h_cs+P1h_ss
        return hm_preds, n_ts
