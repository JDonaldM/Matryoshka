# Matryoshka
A Python package for predicting the galaxy power spectrum with a neural network (NN) based emulator.

## Installation

The package can be installed by cloning this repository and using pip.

```
git clone https://github.com/JDonaldM/Matryoshka
cd Matryoshka
pip install -e .
```

## Basic usage

The example bellow shows how to generate a prediction for a Planck18 LCDM transfer function using `matryoshka`.

```python
import numpy as np
import matryoshka.emulator as Matry
from astropy.cosmology import Planck18_arXiv_v2

COSMO = np.array([Planck18_arXiv_v2.Om0, Planck18_arXiv_v2.Ob0, Planck18_arXiv_v2.H0.value/100,
                  Planck18_arXiv_v2.Neff, -1.0])

TransferEmu = Matry.Transfer()

EmuPred = TransferEmu.emu_predict(COSMO)
```

For more examples and full documentation see https://matryoshka-emu.readthedocs.io/en/latest/

## New in v0.1.0

In the most recent version of `matryoshka` we have included an emulator to predict the nonlinear boost for the *matter* power spectrum that has been trained on the [Quijote simulations](https://arxiv.org/abs/1909.05273). We also include a version of the transfer function emulator that has been trained on the Quijote sample space.

## A note about the nonlinear boost component emulator

In the current version of `matryoshka` the nonlinear boost component emulator has only been trained with training data generated with [HALOFIT](https://iopscience.iop.org/article/10.1088/0004-637X/761/2/152) and serves to demonstrate the use of `matryoshka`. Future versions will include a nonlinear boost component emulator trained with data produced with high resolution N-body simulatios.

## License & Attribution

Copyright 2021 Jamie Donald-McCann. `matryoshka` is free to use under the MIT license, if you find it useful for your research please cite [Donald-McCann et al. (2021)](https://arxiv.org/abs/2109.15236).
