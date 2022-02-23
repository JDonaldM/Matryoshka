===================
matryoshka.emulator
===================

=====================
Galaxy power spectrum
=====================

-------------------
EFTofLSS multipoles
-------------------

This set of six component emulators, collectively reffered to as the ``EFTEMU``, apprxoimates the galaxy power spectrum multipoles that would be predicted by the EFTofLSS code `PyBird <https://github.com/pierrexyz/pybird>`_. See `arXiv:2202.07557 <https://arxiv.org/abs/2202.07557>`_ for details.

.. autoclass:: matryoshka.emulator.EFT
    :members:

-------------------------
Halo model power spectrum
-------------------------

This set of component emulators predictcs the halo model power spectrum. A nonlinear boost can be applied to this power spectrum, as in `arXiv:2109.15236 <https://arxiv.org/abs/2109.15236>`_ (it should be noted that the nonlinear boost emulator is currently trained using HALOFIT, a version trained using simulations is coming). It is also possible to use nonlinear predictions for the dark matter power spectrum here, see ``QUIP`` below. 

.. autoclass:: matryoshka.emulator.HaloModel     
    :members:

==========================
Dark matter power spectrum
==========================

The QUIjote matter Power spectrum emulator, or ``QUIP``. This set of emulators has been trained to on the `Qujote simulations <https://quijote-simulations.readthedocs.io/en/latest/>`_ to predict the nonlinear matter power spectrum.

.. autoclass:: matryoshka.emulator.QUIP
    :members:

====================================
Complete list of component emulators
====================================

.. autoclass:: matryoshka.emulator.Transfer
    :members:

.. autoclass:: matryoshka.emulator.Sigma
    :members:

.. autoclass:: matryoshka.emulator.SigmaPrime   
    :members:

.. autoclass:: matryoshka.emulator.Growth
    :members:

.. autoclass:: matryoshka.emulator.Boost
    :members:

.. autoclass:: matryoshka.emulator.MatterBoost
    :members:

.. autoclass:: matryoshka.emulator.P11l
    :members:

.. autoclass:: matryoshka.emulator.Ploopl
    :members:

.. autoclass:: matryoshka.emulator.Pctl
    :members:
