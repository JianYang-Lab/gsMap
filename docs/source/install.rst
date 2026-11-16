.. _installation:

Installation guide
==================


The ``gsMap`` package can be installed via pip:

.. code-block:: bash

    conda create -n gsMap python>=3.10
    conda activate gsMap
    pip install "gsMap<2"

or via the source code:

.. code-block:: bash

    git clone --branch 1.x https://github.com/JianYang-Lab/gsMap
    cd gsMap
    pip install -e .

Verify the installation by running the following command:

.. code-block:: bash

    gsmap -v
