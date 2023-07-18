Getting Started with HBMep
=================================

Hierarchical Bayesian framewok for modeling Motor Evoked Potential Size.

What is hbmep?
------------------


A Simple Example
------------------
Begin by creating a TOML file, :file:`config.toml`, and enter the following content.

.. code-block:: toml

    [paths]
    csv = "simulation"
    build_dir = "/home/vishu/reports/simulation/"

    [vars]
    subject = "participant"
    features = ["compound_position"]
    intensity = "pulse_amplitude"
    response = ["biceps_auc", "triceps_auc"]

    [mcmc]
    chains = 4
    warmup = 4000
    samples = 6000

    [model]
    link = "rectified_logistic"

    [rectified_logistic]
    "µ_a" = [150, 20]
    "σ_a" = 20
    "σ_b" = 0.1
    "σ_L" = 0.05
    "σ_H" = 5
    "σ_v" = 5
    "p" = 10
    "g_1" = 20
    "g_2" = 20



- ``csv`` points to the dataset in csv format. Setting this to "simulation" will simulate data from the chosen model.
- ``build_dir`` build directory where HBMep will store model artefacts, such as recruitment curve plots. If this does not already exists, HBMep will create it.
- ``subject``, ``features``, ``intensity`` and ``response`` together define/constitute the metadata.
- ``[mcmc]`` configures the NUTS sampler.
- ``[model]`` determines the link function for modeling the recruitment curves.
- ``[rectified_logistic]`` defines priors for the chosen link function.

Installation
^^^^^^^^^^^^^^
TBA

.. note::

    ``hbmep`` installation requrires Python>=3.11. Replace ``python`` with a compatible version, for e.g., ``python3.11``

Tutorial
^^^^^^^^^

.. tip::

   If you have trouble running the commands in this tutorial, please copy the command
   and its output, then `open an issue`_ on the `hbmep`_ repository on
   GitHub. We'll do our best to help you!

.. code-block:: python

    from hbmep.config import Config
    from hbmep.model import Model

.. code-block:: python

    # Initalize config
    config = Config(toml_path=toml_path)

    # Initalize model
    model = Model(config=config)

.. code-block:: python

    # Load data
    df, encoder_dict = model.load()

    # Run inference
    mcmc, posterior_samples = model.run_inference(df=df)

    # Plot recruitment curves
    model.render_recruitment_curves(
        df=df,
        encoder_dict=encoder_dict,
        posterior_samples=posterior_samples
    )


Supplemental
-------------
.. code-block:: python

    # Plot dataset
    model.plot(df=df, encoder_dict=encoder_dict)

    # Plot posterior predictive check
    model.render_predictive_check(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples)

Inspecting the model
^^^^^^^^^^^^^^^^^^^^^
A summary with the parameter estimates and their uncertainties can be generated using the ``diagnostics`` method.

.. code-block:: python

    # Convergence diagnostics
    model.diagnostics(mcmc=mcmc)

Posterior Predictive Check
^^^^^^^^^^^^^^^^^^^^^^^^^^^
If the model is any good, data simulated from it should be pretty similar to the data actually observed.

.. code-block:: python

    # Plot posterior predictive check
    model.render_predictive_check(
        df=df,
        encoder_dict=encoder_dict,
        posterior_samples=posterior_samples
    )

Setting Priors
^^^^^^^^^^^^^^^^^
Add some note here

.. code-block:: python

    # Plot prior predictive check
    model.render_predictive_check(
        df=df,
        encoder_dict=encoder_dict
    )
