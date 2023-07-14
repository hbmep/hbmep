Configuration File
=========================

HBMep uses a TOML configuration file for inference. It tells HBMep about the dataset which is to be used for inference, the parameters for the NUTS sampler, and the link function for modeling recruitment curves.

This tutorial walks you through the different components of the configuration file.

.. tip::

   If you have trouble running the commands in this tutorial, please copy the command
   and its output, then `open an issue`_ on the `packaging-problems`_ repository on
   GitHub. We'll do our best to help you!

.. _open an issue: https://github.com/pypa/packaging-problems/issues/new?template=packaging_tutorial.yml&title=Trouble+with+the+packaging+tutorial&guide=https://packaging.python.org/tutorials/packaging-projects

.. _packaging-problems: https://github.com/pypa/packaging-problems

Creating config.toml
-----------------------
Begin by creating a TOML file, and for convenience, we will call it :file:`config.toml`


Configuring dataset
^^^^^^^^^^^^^^^^^^^^

Open :file:`config.toml` and enter the following content. Change the ``csv``
to point to your dataset.

.. code-block:: toml

    [paths]
    csv = "/home/user/data/mock.csv"
    build_dir = "/home/user/hbmep-artefacts/test_run_01"

    [vars]
    subject = "participant"
    features = ["compound_position"]
    intensity = "pulse_amplitude"
    response = ["biceps_auc", "triceps_auc"]

- ``csv`` points to the dataset in csv format.
- ``build_dir`` build directory where HBMep will store model artefacts, such as recruitment curve plots. If this does not already exists, HBMep will create it.
- ``subject`` dataset column for the names or indices of test subjects.
- ``features`` TBA
- ``intensity`` dataset column for the stimulation intensities at which MEPs are observed.
- ``response`` list of dataset columns for the observed MEPs.

See the :ref:`MEP Data Specification <declaring-mep-data>` for
declating the optional motor evoked potential data in the ``[optional.mep_data]``
table.

Configuring sampler
^^^^^^^^^^^^^^^^^^^^

The ``[mcmc]`` table can be used to configure the NUTS sampler.

.. code-block:: toml

    [mcmc]
    chains = 4
    warmup = 4000
    samples = 6000

- ``chains`` number of MCMC chains to run in parallel.
- ``warmup`` number of warmup or burn-in samples.
- ``samples`` number of samples to generate from the Markov chains post warmup.

.. tip::

    (here, add some note to hint that these settings can be left as they are in the template)

Configuring model
^^^^^^^^^^^^^^^^^^^^

The ``[model]`` table is used to select the link function for the parametric form of recruitment curves. Priors for the chosen model are set under its corresponding table.

.. code-block:: toml

    [model]
    link = "rectified_logistic"

    [rectified_logistic]
    "µ_a" = [150, 20]
    "σ_a" = 20
    "σ_b" = 0.5
    "σ_L" = 0.05
    "σ_H" = 50
    "σ_v" = 5
    "g_1" = 20
    "g_2" = 20
    "p" = 10

    [saturated_relu]
    "µ_a" = [150, 50]
    "σ_a" = 50
    "σ_b" = 0.1
    "σ_L" = 0.05
    "σ_H" = 5
    "σ_v" = 10

    [relu]
    "µ_a" = [150, 50]
    "σ_a" = 50
    "σ_b" = 0.1
    "σ_L" = 0.05
    "σ_H" = 5
    "σ_v" = 10
    "g_1" = 20
    "g_2" = 20

- ``link`` parametric form of recruitment curves. Must be one of "rectified_logistic", "saturated_relu", or "relu".

See the :ref:`Link Functions and Priors <link-functions>` for details on the available parametric forms and setting their priors.


Link Functions and Priors
-------------------------


MEP Data Specification
----------------------