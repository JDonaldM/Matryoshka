# Matryoshka
A Python package for predicting the galaxy power spectrum with a neural network (NN) based emulator.

## Installation

The package can be installed by cloning this repository and using pip.

```
cd path\to\Matryoshka
pip install .
```

## Basic usage

```python
import numpy as np
import matryoshka.emulator as Matry
from astropy.cosmology import Planck18_arXiv_v2

COSMO = np.array([Planck18_arXiv_v2.Om0, Planck18_arXiv_v2.Ob0, Planck18_arXiv_v2.H0.value/100,
                  Planck18_arXiv_v2.Neff, -1.0])

TransferEmu = Matry.Transfer()

EmuPred = TransferEmu.emu_predict(COSMO, single_or_batch="single", mean_or_full="mean")
```
