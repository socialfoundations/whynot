.. _estimators:

Provided Estimators
===================

WhyNot comes equipped with a collection of estimators to enable rapid
comparisons of methods on new benchmarks and with new experimental designs.
We hope this both allows users of causal inference tools to compare methods and
choose methods that perform well in situations of interest. We also hope that
WhyNot will allow producers of new causal inference methods to rapidly evaluate
their algorithms against a rich set of baselines on common datasets.


Beyond estimators provided by WhyNot, the data generated in :class:`~whynot.framework.Dataset`
is easily accessible as NumPy arrays that can be feed into the user's own
estimators.

.. code:: python

    >>> import whynot as wn

    >>> dset = wn.world3.PollutionRCT.run(num_samples=200)
    >>> covariates, treatment, outcomes = dset.covariates, dset.treatments, dset.outcomes
    
    # Replace with your favorite estimator!
    >>> estimate = wn.algorithms.ols.estimate_treatment_effect(covariates, treatment, outcomes)

.. _ate-estimators:

Average Treatment Effect Estimators
-----------------------------------

* Linear Regression
* Matching, using the package `Matching <http://sekhon.berkeley.edu/matching/>`_.
* IP Weighting, using the package `WeightIt  <https://github.com/ngreifer/WeightIt>`_.
* Causal Forests, using the package `GRF <https://github.com/grf-labs/grf>`_.
* Causal Bart, using the package `BartCause <https://github.com/vdorie/bartCause>`_.
* TMLE, using the package `TMLE <https://cran.r-project.org/package=tmle>`_.
* Double machine learning, using the package `econml <https://github.com/microsoft/EconML>`_.

By default, linear regression and propensity score matching are included with
WhyNot. To minimize dependencies for the original package, the remaining
estimators ship with the companion `WhyNot-Estimators <https://github.com/zykls/whynot_estimators>`_ package.
For more installation details, see :ref:`installation`.

.. _hte-estimators:

Heterogeneous Treatment Effect Estimators
-----------------------------------------

* Causal Forests, using the package `GRF <https://github.com/grf-labs/grf>`_.
* Causal Bart, using the package `BartCause <https://github.com/vdorie/bartCause>`_.
* Double machine learning, using the package `econml <https://github.com/microsoft/EconML>`_.
