import numpy as np
import pytest


class TestScalers:

    # Generate some random data.
    data = np.random.randn(1000,5)

    # Manually compute the parameters of the scalers.
    min_val = data.min(axis=0)
    diff = data.max(axis=0)-data.min(axis=0)
    mean = data.mean(axis=0)
    stddev = data.std(axis=0)
    min_val_log = np.log(np.abs(data)).min(axis=0)
    diff_log = np.log(np.abs(data)).max(axis=0)-np.log(np.abs(data)).min(axis=0)

    def test_UnifromScaler(self):
        from matryoshka.training_funcs import UniformScaler

        # Manually transform the data.
        data_prime = (self.data-self.min_val)/self.diff

        # Initalise scaler.
        scaler = UniformScaler()

        # Make sure error is raised if data isnt 2d.
        with pytest.raises(ValueError):
            scaler.fit(self.data[0])

        # Fit the data.
        scaler.fit(self.data)

        # Check the scaler parameters.
        assert np.all(scaler.min_val  == self.min_val)
        assert np.all(scaler.diff == self.diff)

        # Check transformed data.
        assert np.all(scaler.transform(self.data) == data_prime)

        # Check inverse transform.
        assert np.all(scaler.inverse_transform(scaler.transform(self.data)) == (data_prime*self.diff)+self.min_val)

    def test_LogScaler(self):
        from matryoshka.training_funcs import LogScaler

        # Manually transform the data.
        data_prime = (np.log(np.abs(self.data))-self.min_val_log)/self.diff_log

        # Initalise scaler.
        scaler = LogScaler()

        # Make sure error is raised if data isnt 2d.
        with pytest.raises(ValueError):
            scaler.fit(self.data[0])

        # Make sure an error is raised if negative data is passed.
        with pytest.raises(ValueError):
            scaler.fit(self.data)

        # Fit the |data|.
        scaler.fit(np.abs(self.data))

        # Check the scaler parameters.
        assert np.all(scaler.min_val  == self.min_val_log)
        assert np.all(scaler.diff == self.diff_log)

        # Check transformed data.
        assert np.all(scaler.transform(np.abs(self.data)) == data_prime)

        # Check inverse transform.
        assert np.all(scaler.inverse_transform(scaler.transform(np.abs(self.data)))\
               == np.exp((data_prime*self.diff_log)+self.min_val_log))

    def test_StandardScaler(self):
        from matryoshka.training_funcs import StandardScaler

        # Manually transform the data.
        data_prime = (self.data-self.mean)/self.stddev

        # Initalise scaler.
        scaler = StandardScaler()

        # Make sure error is raised if data isnt 2d.
        with pytest.raises(ValueError):
            scaler.fit(self.data[0])

        # Fit the data.
        scaler.fit(self.data)

        # Check the scaler parameters.
        assert np.all(scaler.mean  == self.mean)
        assert np.all(scaler.scale == self.stddev)

        # Check transformed data.
        assert np.all(scaler.transform(self.data) == data_prime)

        # Check inverse transform.
        assert np.all(scaler.inverse_transform(scaler.transform(self.data)) == (data_prime*self.stddev)+self.mean)