.. _installation:

Installation
=================================
.. note::

    hbMEP requires Python >= 3.11. We recommend using Python 3.11, so replace ``python`` with ``python3.11``.

On Linux / MacOS
----------------
The recommended way to install hbMEP is to create a new virtual environment and install the package in it with `pip <https://pip.pypa.io/>`_. This way, you can avoid conflicts with other packages that you may have installed on your system.

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    pip install hbmep

If you don't have a compatible version of Python installed, you can use `conda <https://conda.io>`_ to create a new environment with the required version of Python.

.. code-block:: bash

    conda create -n python-311 python=3.11 -y
    conda activate python-311
    python -m venv .venv
    conda deactivate

    source .venv/bin/activate
    pip install hbmep

Now, the Python interpreter should be located at ``.venv/bin/python``. You can use this to run scripts that make use of hbMEP.

On Windows
----------------
Use conda to create a new environment with the required version of Python.

.. code-block:: bash

    conda create -n python-311 python=3.11 -y
    conda activate python-311
    python -m venv .venv
    conda deactivate

Now, activate the virtual environment and install hbMEP.

.. code-block:: powershell

    .venv\Scripts\activate
    pip install hbmep
