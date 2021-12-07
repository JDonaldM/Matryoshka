Conveniently Re-train the ``EFTEMU``
====================================

We include the convenience script ``trainEFTEMUcomponents.py`` so that it is very simple to re-train the ``EFTEMU`` if desired.

To use the script you will need to use the directory structure shown below for your new training data::

    ./new_data
    +-- features
        +-- cosmos1.npy
        +-- cosmos2.npy
        +-- ...
    +-- functions
        +-- P110
            +-- P110_1.npy
            +-- P110_2.npy
            +-- ...
        +-- P112
            +-- P112_1.npy
            +-- ...
        +-- Ploop0
            +-- Ploop0_1.npy
            +-- ...
        +-- Ploop2
            +-- ...
        +-- Pct0
            +-- Pct0_1.npy
            +-- ...
        +-- Pct2
            +-- ...

Where the ``cosmosX.npy`` files contain arrays of shape ``(nsamp, nparam)``, and the ``PY_X.npy`` files contain the component predictions corresponding to the cosmological parameters in the ``cosmosX.npy`` files made with ``PyBird``.

Once you have your new data in the correct structure you can re-train the emulator with the following command.

.. code:: bash

    python trainEFTEMUcomponents.py --inputX $path_to_features --inputY $path_to_functions --cache $path_to_matryoshkadata

All posible arguments can be listed with the following command.

.. code:: bash

    python trainEFTEMUcomponents.py -h

It should be noted that tis script only allows for very limited adjustment of the NNs that form each of the component emulators. If you do not get good results using the script try creating your own using the one provided as a template and adjust some of the hyperparameters that enter into the ``trainNN`` function.

 
