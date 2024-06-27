.. _installation:

Installation
=================================
.. note::

    ``hbmep`` installation requrires Python>=3.11. Replace ``python`` with a compatible version, for e.g., ``python3.11``

On Linux / MacOS
----------------
The recommended way to install `hbmep` is to create a new virtual environment and install the package in it with `pip <http://www.pip-installer.org/>`_. This way, you can avoid conflicts with other packages that you may have installed in your system.

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    pip install hbmep

Note that you can replace ``.venv`` with any other directory name that you prefer. This is where the virtual environment will be created.

If you don't have a compatible version of Python installed, you can use `conda <https://conda.io>`_ to create a new environment with the required version of Python.

.. code-block:: bash

    conda create -n python-311 python=3.11 -y
    conda activate python-311
    python -m venv .venv
    conda deactivate

    source .venv/bin/activate
    pip install hbmep

Now, the Python interpreter should be located at ``.venv/bin/python``. You can use this to run your scripts that make use of `hbmep`.

On Windows
----------------
Use conda to create a new environment with the required version of Python.

.. code-block:: bash

    conda create -n python-311 python=3.11 -y
    conda activate python-311
    python -m venv .venv
    conda deactivate

Again, you can replace ``.venv`` with any other directory name that you prefer. This is where the virtual environment will be created.

Now, activate the virtual environment and install `hbmep`.

.. code-block:: powershell

    .venv\Scripts\activate
    pip install hbmep
