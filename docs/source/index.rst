.. matryoshka documentation master file, created by
   sphinx-quickstart on Tue Sep  7 18:04:02 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to matryoshka's documentation!
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


``matryoshka`` is a Python package for predicting the galaxy power spectrum with a neural network (NN) based emulator.

Installation
============

The package can be installed by cloning the repository and using ``pip``.::


	git clone https://github.com/<your-username>/Matryoshka.git
	cd path\to\Matryoshka
	pip install .


The repository contains the directory ``matryoshka-data`` which contains all the weights for the NNs along with the data used to train them.

Basic Usage
===========

The example below shows how to generate a prediction for a Planck18 LCDM transfer function using ``matryoshka``.

.. code:: Python

	import numpy as np
	import matryoshka.emulator as Matry
	from astropy.cosmology import Planck18_arXiv_v2
	
	COSMO = np.array([Planck18_arXiv_v2.Om0, Planck18_arXiv_v2.Ob0, Planck18_arXiv_v2.H0.value/100,
	                  Planck18_arXiv_v2.Neff, -1.0])
	
	TransferEmu = Matry.Transfer()
	
	EmuPred = TransferEmu.emu_predict(COSMO, mean_or_full="mean")

Setting ``mean_or_full="mean"`` results in the ensemble mean prediction being returned rather than predictions from each ensemble member.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

.. toctree::
    :maxdepth: 3
    :hidden:

    api
    examples
