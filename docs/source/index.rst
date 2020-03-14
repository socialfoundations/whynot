.. image:: _static/WhyNot_fullcolor.svg

WhyNot is a Python package that provides an experimental sandbox for causal
inference and decision making in dynamics. Starting with a suite of dynamic
simulations that present realistic technical challenges, WhyNot makes it easy
for researchers to develop, test, and benchmark methods for causal inference and
reinforcement learning.

What does this look like? The following code generates an observational dataset
using an HIV treatment simulator and then runs linear regression to estimate the
average treatment effect. These estimates can then subsequently be benchmarked
against ground truth.

.. code:: python
    
    import numpy as np
    import whynot as wn

    # Construct causal experiments on sophisticated computer simulations
    experiment = wn.hiv.HIVConfounding

    # Generate an observational dataset
    dataset = experiment.run(num_samples=200, show_progress=True)

    # Run your favorite causal inference procedure
    estimate = wn.algorithms.ols.estimate_treatment_effect(
        dataset.covariates, dataset.treatments, dataset.outcomes)

    # Benchmark the average treatment effect results against the
    # ground truth in the dataset
    ate = np.mean(dataset.true_effects)
    relative_error = np.abs((estimate.ate - ate) / ate)



WhyNot also supports benchmarking and investigation of causal inference tools
for heterogenous treatment effects and causal graph discovery. Beyond causal
inference, WhyNot provides simulators and environments to study decision making
in dynamics, both in the context of reinforcement learning, as well as from
recent perspectives like delayed impact, strategic classification, and
performative prediction.


Documentation
-------------

This part of the documentation guides you through all of the library's
usage patterns.
    
.. toctree::
   :maxdepth: 2

   why
   installation
   quickstart
   simulators
   usage
   examples
   develop

API Reference
-------------

If you are looking for information on a specific function, class, or
method, this part of the documentation is for you.

.. toctree::
   :maxdepth: 2

   api
