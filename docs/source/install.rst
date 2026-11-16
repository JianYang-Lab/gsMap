.. _installation:

Installation guide
==================

.. warning::

   **gsMap 1.x is deprecated.** This documentation describes gsMap 1.x (1.73.x), which
   receives bug fixes only. Pin the version when installing:
   ``pip install "gsMap<2"`` or ``conda install "bioconda::gsmap<2"``. Without a version,
   ``pip install gsMap`` installs gsMap 2.x on Python 3.12 or later.

   New analyses should use gsMap 2.x: see the `gsMap 2.x documentation <https://jianyang-lab.github.io/gsMap/>`_ and
   `Upgrading to gsMap 2.0 <https://jianyang-lab.github.io/gsMap/upgrading.html>`_.



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
