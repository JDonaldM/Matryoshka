# Matryoshka
A Python package for predicting the galaxy power spectrum with a neural network (NN) based emulator.

## Installation

The package can be installed by cloning this repository and using pip.

```
git clone https://github.com/JDonaldM/Matryoshka
cd Matryoshka
pip install .
```

## Basic usage

The example bellow shows how to generate a prediction for a Planck18 LCDM transfer function using `matryoshka`. Setting `mean_or_full="mean"` results in the ensemble mean prediction being returned rather than predictions from each ensemble member.

```python
import numpy as np
import matryoshka.emulator as Matry
from astropy.cosmology import Planck18_arXiv_v2

COSMO = np.array([Planck18_arXiv_v2.Om0, Planck18_arXiv_v2.Ob0, Planck18_arXiv_v2.H0.value/100,
                  Planck18_arXiv_v2.Neff, -1.0])

TransferEmu = Matry.Transfer()

EmuPred = TransferEmu.emu_predict(COSMO, mean_or_full="mean")
```

For more examples and full documentation see https://matryoshka-emu.readthedocs.io/en/latest/

## A note about the nonlinear boost component component emulator

In the current version of `matryoshka` the nonlinear boost component emulator has only been trained with training data generated with [HALOFIT](https://iopscience.iop.org/article/10.1088/0004-637X/761/2/152) and serves to demonstrate the use of `matryoshka`. Future versions will include a nonlinear boost component emulator trained with data produced with high resolution N-body simulatios.
