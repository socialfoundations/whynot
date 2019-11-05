.. image:: _static/WhyNot_fullcolor.svg

WhyNot is a Python package that provides an experimental sandbox for causal
inference. Starting with a suite of dynamic simulations that present realistic
technical challenges, WhyNot makes it easy for researchers to develop, test, and
benchmark causal inference methods.

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
    rel_error = np.abs((estimate.ate - ate) / ate)



WhyNot also supports benchmarking and investigation of causal inference tools
for heterogenous treatment effects, causal graph discovery, and sequential
decision making.


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
   estimators
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
