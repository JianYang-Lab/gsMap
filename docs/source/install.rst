.. _installation:

Installation
============

**gsMap** 2.x requires Python 3.12 or later and is currently supported on Linux.
Windows compatibility has not been verified, so Windows installation instructions are
not provided.

gsMap uses both **PyTorch** and **JAX**. Select exactly one optional dependency for the
compute backend so that both libraries use compatible runtimes. Do not combine backend
extras or use ``--all-extras``.

Choose a compute backend
------------------------

.. list-table:: Backend selection
   :header-rows: 1
   :widths: 18 24 24 34

   * - Extra
     - PyTorch backend
     - JAX backend
     - Use case
   * - ``cpu``
     - CPU
     - CPU
     - CPU-only installations and continuous integration.
   * - ``cuda126``
     - CUDA 12.6
     - CUDA 12
     - Compatibility profile for CUDA 12 deployments.
   * - ``cuda128``
     - CUDA 12.8
     - CUDA 12
     - Newer CUDA 12 profile.
   * - ``cuda130``
     - CUDA 13.0
     - CUDA 13
     - CUDA 13 profile for compatible systems.
   * - ``tpu``
     - CPU
     - TPU
     - JAX runs on TPU while PyTorch remains on CPU.

The CUDA extra names identify the PyTorch wheel channel. Check the
`PyTorch CUDA support matrix
<https://github.com/pytorch/pytorch/blob/main/RELEASE.md#pytorch-cuda-support-matrix>`_
and the `JAX installation guide
<https://docs.jax.dev/en/latest/installation.html>`_ when selecting a backend for a
particular system.

Install from PyPI
-----------------

Install a released package with one compute backend. ``uv``'s
``--torch-backend`` selects the matching official PyTorch wheel index. For an
editable installation or the dataset-preparation utilities supplied with the
repository, use **Install from source**.

.. note::

   gsMap 2.0 is in beta. The commands below pin the pre-release ``2.0.0b1``; without
   a version, ``pip`` and ``uv`` keep installing gsMap 1.x until 2.0.0 is released.

Install ``uv`` and create the environment:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv --python 3.13
   source .venv/bin/activate

Install exactly one backend:

.. code-block:: bash

   # Choose exactly one command.
   uv pip install "gsMap[cpu]==2.0.0b1" --torch-backend=cpu
   uv pip install "gsMap[cuda126]==2.0.0b1" --torch-backend=cu126
   uv pip install "gsMap[cuda128]==2.0.0b1" --torch-backend=cu128
   uv pip install "gsMap[cuda130]==2.0.0b1" --torch-backend=cu130
   uv pip install "gsMap[tpu]==2.0.0b1" --torch-backend=cpu

``--torch-backend`` is an ``uv pip`` option. It ensures that PyTorch comes from the
selected wheel channel while gsMap and JAX are installed from PyPI.

Create an environment with Conda
--------------------------------

Conda can manage the Python environment while ``uv`` installs the released package and
selects its binary backend:

.. code-block:: bash

   conda create --name gsmap python=3.13 pip
   conda activate gsmap
   python -m pip install uv

   # Choose exactly one command.
   uv pip install "gsMap[cpu]==2.0.0b1" --torch-backend=cpu
   uv pip install "gsMap[cuda126]==2.0.0b1" --torch-backend=cu126
   uv pip install "gsMap[cuda128]==2.0.0b1" --torch-backend=cu128
   uv pip install "gsMap[cuda130]==2.0.0b1" --torch-backend=cu130
   uv pip install "gsMap[tpu]==2.0.0b1" --torch-backend=cpu

This approach uses Conda only to isolate Python. Installing the project with ``uv`` keeps
the backend selection identical to the direct PyPI instructions.

Install from source
-------------------

Clone the repository and use ``uv sync``. The repository lockfile and
conditional package indexes make each backend reproducible:

.. code-block:: bash

   git clone https://github.com/JianYang-Lab/gsMap.git
   cd gsMap
   git rev-parse HEAD

   # Choose exactly one command.
   uv sync --locked --python 3.13 --extra cpu
   uv sync --locked --python 3.13 --extra cuda126
   uv sync --locked --python 3.13 --extra cuda128
   uv sync --locked --python 3.13 --extra cuda130
   uv sync --locked --python 3.13 --extra tpu

``uv sync`` creates ``.venv``, installs gsMap in editable mode, and applies ``uv.lock``.
``--locked`` checks that the lockfile matches project metadata, without silently
updating it; see the `uv sync documentation
<https://docs.astral.sh/uv/concepts/projects/sync/>`_. Activate ``.venv`` before
running tutorials. Run the displayed ``gsmap`` commands or Python API calls
from your own analysis directory. Tutorial dataset archives include the required
input lists and GWAS configuration files; preparation scripts are for maintainers.
Verify the selected accelerator on the machine where you will run the analysis.
The default development group includes the test dependencies. Add ``--no-dev`` for a
runtime-only environment or ``--group doc`` when building the documentation.

An exact sync removes packages that are not part of the selected project environment. Use
a dedicated environment if unrelated software stacks must be retained. To sync an already
activated environment instead of ``.venv``, use ``--active``:

.. code-block:: bash

   source /path/to/environment/bin/activate
   uv sync --active --extra cuda126

Upgrading from gsMap 1.x
------------------------

gsMap 1.x users should read :doc:`upgrading` for the renamed commands and new reference
resources. To keep using gsMap 1.x, install ``"gsMap<2"``; its source is on the
`1.x branch <https://github.com/JianYang-Lab/gsMap/tree/1.x>`_.

Verify the installation
-----------------------

Run accelerator checks inside a session that has access to the selected device:

.. code-block:: bash

   python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
   python -c "import jax; print(jax.__version__, jax.devices())"
   gsmap --version

For a CUDA installation, also run a small operation in each framework:

.. code-block:: bash

   python -c "import torch; print(torch.ones(1, device='cuda'))"
   python -c "import jax.numpy as jnp; print(jnp.ones(1).block_until_ready())"

Troubleshooting
---------------

CUDA kernel compatibility errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If PyTorch reports ``no kernel image is available for execution on the device``, the
installed wheel does not include kernels for the available GPU architecture. Select a
compatible backend using the upstream support matrix. The CUDA 12.6 compatibility profile
can be installed from a source checkout with:

.. code-block:: bash

   uv sync --extra cuda126

Compiler path errors while building ``pandas-plink``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``CC`` or ``CXX`` points to a compiler that no longer exists, clear the stale overrides
for the sync so the system compiler is used:

.. code-block:: bash

   env -u CC -u CXX uv sync --extra cuda126

Timeout while downloading a large PyTorch wheel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA-enabled PyTorch wheels are large. On a slow connection, increase uv's HTTP timeout:

.. code-block:: bash

   UV_HTTP_TIMEOUT=600 uv sync --extra cuda126

JAX cannot initialize CUDA
~~~~~~~~~~~~~~~~~~~~~~~~~~

Confirm that the installed NVIDIA driver supports the selected CUDA backend using the JAX
installation guide linked above. Also check that ``LD_LIBRARY_PATH`` is not overriding the
CUDA and cuDNN libraries installed by the wheels.
