Conveniently Re-train the ``EFTEMU``
====================================

We include two convenience scripts, ``genEFTEMUtraindata.py`` and ``trainEFTEMUcomponents.py``, so that it is very simple to re-train the ``EFTEMU`` if desired.

The first script, ``genEFTEMUtraindata.py``, calculates the bias indpendent components with ``CLASS`` and ``PyBird``.

.. code:: bash

    scripts]$ python genEFTEMUtraindata.py -h
    usage: genEFTEMUtraindata.py [-h] --inputX INPUTX --save_dir SAVE_DIR
                                --redshift REDSHIFT [--optiresum OPTIRESUM]

    optional arguments:
    -h, --help            show this help message and exit
    --inputX INPUTX       Directroy with files containg the training
                            cosmologies.
    --save_dir SAVE_DIR   Path to save outputs.
    --redshift REDSHIFT   Redshift at which to generate the data.
    --optiresum OPTIRESUM
                            Boolean. Use pybird optimal resummation. Can be 1 or
                            0.

This script will save the computed bias indpendent terms with the following data structure::

    +-- save_dir
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

This is the structure expected by ``trainEFTEMUcomponents.py`` so bare this in mind if you decide to generate new training data without the ``genEFTEMUtraindata.py`` script.

Once you have your new data in the correct structure you can re-train the emulator with the ``trainEFTEMUcomponents.py`` script.

.. code:: bash

    scripts]$ python trainEFTEMUcomponents.py -h
    usage: trainEFTEMUcomponents.py [-h] --inputX INPUTX --inputY INPUTY --cache
                                    CACHE [--new_split NEW_SPLIT]
                                    [--archP110 ARCHP110] [--archP112 ARCHP112]
                                    [--archPloop0 ARCHPLOOP0]
                                    [--archPloop2 ARCHPLOOP2]
                                    [--archPct0 ARCHPCT0] [--archPct2 ARCHPCT2]
                                    [--verbose VERBOSE] [--to_train TO_TRAIN]

    optional arguments:
    -h, --help            show this help message and exit
    --inputX INPUTX       Directory with feature files.
    --inputY INPUTY       Directory with target function files.
    --cache CACHE         Path to save outputs.
    --new_split NEW_SPLIT
                            Use a new train test split? 0 for no, 1 for yes.
                            Default 0.
    --archP110 ARCHP110   Architecture for P110 emulator. pass as a string i.e.
                            '200 200'. This specifies two hidden layers with 200
                            nodes each. Default '200 200'.
    --archP112 ARCHP112   Architecture for P112 emulator. Default '200 200'.
    --archPloop0 ARCHPLOOP0
                            Architecture for Ploop0 emulator. Default '400 400'.
    --archPloop2 ARCHPLOOP2
                            Architecture for Ploop2 emulator. Default '400 400'.
    --archPct0 ARCHPCT0   Architecture for Pct0 emulator. Default '200 200'.
    --archPct2 ARCHPCT2   Architecture for Pct2 emulator. Default '200 200'.
    --verbose VERBOSE     Verbose for tensorflow. Default 0.
    --to_train TO_TRAIN   Componenets to train. Pass as a string i.e. 'Ploop
                            Pct'. This will only train the Ploop and Pct
                            components. Default 'P11 Ploop Pct'.

Setting the variable ``--cache`` to the full path to ``matryoshka-data/EFTv2/redshift/`` will mean that no modification to ``matryoshka`` need to be made to use your newly trained emulator.
It is also possible to save your new emulator as a new ``version``. To do this set ``--cache`` to to the full path to ``matryoshka-data/EFTv3/redshift`` for example. Your new version can then be used by specifying it when initalising the emulator:

.. code:: Python

	import matryoshka.emulator as MatEmu
	
	P0_emu = MatEmu.EFT(0, version="EFTv3", redshift=redshift)

It should be noted that the ``trainEFTEMUcomponents.py`` script only allows for very limited adjustment of the NNs that form each of the component emulators. If you do not get good results using the script try creating your own using the one provided as a template and adjust some of the hyperparameters that enter into the ``trainNN`` function.
