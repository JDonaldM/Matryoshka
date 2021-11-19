===================
matryoshka.emulator
===================

Module for producing predictions with the component emulators of ``matryoshka``.

Base Model
==========

.. autoclass:: matryoshka.emulator.Transfer
    :members:

.. autoclass:: matryoshka.emulator.Sigma
    :members:

.. autoclass:: matryoshka.emulator.SigmaPrime   
    :members:

.. autoclass:: matryoshka.emulator.Growth
    :members:

Nonlinear Boost (Matter)
========================

.. autoclass:: matryoshka.emulator.MatterBoost
    :members:

Nonlinear Boost (Galaxy)
========================

In the current version of ``matryoshka`` the nonlinear boost component emulator has only been trained with training data generated with HALOFIT and serves to demonstrate the use of ``matryoshka``. Future versions will include a nonlinear boost component emulator trained with data produced with high resolution N-body simulatios.

.. autoclass:: matryoshka.emulator.Boost
    :members:

Galaxy Power Spectrum
=====================

.. autoclass:: matryoshka.emulator.HaloModel     
    :members:

EFT
===

This emulator apprxoimates the galaxy power spectrum multipoles that would be predicted by the EFTofLSS code `PyBird <https://github.com/pierrexyz/pybird>`_.
In the current version predictions can only be made at z=0.51.

.. autoclass:: matryoshka.emulator.EFT
    :members: