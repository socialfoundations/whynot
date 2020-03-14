
Treatment Effects
=================

.. _average-treatment-effects:

Average Treatment Effects
-------------------------
In the simplest setting, we are interested in estimating the causal effect of a
binary variable :math:`A` on an outcome variable :math:`Y`. Specifically, we
are interested in the *average treatment effect*

.. math::

    \mathrm{ATE} = \mathbb{E}[Y \mid \mathrm{do}(A = 1)] - \mathbb{E}[Y \mid \mathrm{do}(A = 0)].

The mathematical operator :math:`\mathrm{do}(A = a)` denotes an intervention
that holds the value of :math:`A` constant at level :math:`a` (see `Pearl 2009`_
for more details). The average treatment effect can also be defined using the
Neyman-Rubin potential outcomes framework (`Imbens and Rubin 2015`_).
*WhyNot is agnostic to which framework the practitioner applies.*

Due to the possibility of *confounding*, it is not generally possible to
identify the :math:`\mathrm{ATE}` from observations of treatment :math:`A` and
outcome :math:`Y` alone. In some cases, however, when additional measured
*covariates* :math:`X` are available, the :math:`\mathrm{ATE}` is identifiable
(`Pearl 2009`_ or `Imbens and Rubin 2015`_).

WhyNot allows the user to generate observational datasets :math:`(X_i, A_i,
Y_i)_{i=1}^n` consisting of :math:`n` samples of covariates :math:`X_i`,
treatment assignment :math:`A_i`, and observed outcome :math:`Y_i`.
Importantly, WhyNot also also returns the ground truth (sample) average
treatment effect.

**Case-study: Opioid Epidemic Simulation**

The :ref:`opioid-simulator` is a system dynamics model of the US opioid
epidemic. In the model, we wish to study the effect of reducing prescription
opioid use on the total number of opioid overdose death. Concretely, we ask:

    What is the effect of lowering non-medical prescription opioid use by 10%
    in 2015 on the number of opioid overdose deaths in the United States in
    2025?

In our setup, each *unit* is a run of the simulator from a different intial
state. Using the notions developed above, treatment assignment :math:`A`
corresponds to whether or not the run receives the policy intervention to reduce
opioid use in 2015, and the outcome :math:`Y` is the number of overdose deaths
in 2025.

To generate confounding, we imagine governments are more likely to intervene to
reduce opioid abuse if the number of opioid overdose deaths is high.  Therefore,
runs with high levels of opioid overdose deaths in 2015 are more likely to
receive treatment. A sufficient set of covariates :math:`X` is the entire system
state in 2015.

The ``OverdoseConfounding`` experiment implements this logic. The code below
calls the experiment class and generates the causal inference
:class:`~whynot.framework.Dataset` for :math:`n=500` samples.

.. code:: python

    import whynot as wn
    import numpy as np

    overdose_experiment = wn.opioid.Confounding

    dataset  = overdose_experiment.run(num_samples=500)

    # Average over the population
    sample_ate = dataset.sate


One key parameter in the ``Confounding`` experiment is the *strength* of
the confounding. If runs with high levels of treatment get treated with
probability :math:`p` and otherwise get treated with probability :math:`1-p`,
then how does the performance of finite-sample estimators change as :math:`p \to
1`? We can easily generate a sequence of datasets to as :math:`p` varies to
check this since :math:`p` is a :class:`Parameter <whynot.framework.ExperimentParameter>` of the experiment.

.. code:: python

    >>> overdose_experiment = wn.opioid.Confounding
    >>> overdose_experiment.get_parameters()
	Params:
		Name:		nonmedical_incidence_delta
		Description:	Percent decrease in new nonmedical users of prescription opioids.
		Default:	-0.1
		Sample Values:		[-0.073, -0.1]

		Name:		propensity
		Description:	Probability of treatment assignment in high overdose group.
		Default:	0.9
		Sample Values:		[0.5, 0.6, 0.7, 0.9, 0.99]

    >>> datasets = []
    >>> for p in np.arange(0.5, 0.99, 0.05):
    ...     dataset = overdose_experiment.run(num_samples=500, propensity=p)
    ...     datasets.append(dataset)


Estimating Average Treatment Effects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
With a causal dataset in hand, WhyNot provides a collection of estimators in the
:func:`~whynot.causal_suite` to estimate average treatment effects. For a
detailed list of estimators, see :ref:`ate-estimators`. Each estimator returns
an :class:`~whynot.framework.InferenceResult` that includes the estimate of the
ATE, as well as a confidence interval (if provided by the estimator).

.. code:: python

    data = overdose_experiment.run(num_samples=100)
    
    # Estimate ATE using a linear model
    estimate = wn.algorithms.ols.estimate_treatment_effect(data.covariates, data.treatments, data.outcomes)

    # Compare estimate with ground truth
    relative_error = np.abs((estimate.ate - data.sate) / data.sate)


.. _heterogeneous-treatment-effects:

Heterogeneous Treatment Effects
-------------------------------
While average treatment effects are concerned with the causal effect over an
entire population, heterogeneous treatment effects are concerned with the
treatment effect for each individual or for each group defined by covariates
:math:`X=x`. In particular, the *Conditional Average Treatment Effect* (CATE)
for covariates :math:`x` is defined as

.. math::

    \mathrm{CATE}(x) = \mathbb{E}[Y \mid X = x, \mathrm{do}(A = 1)] - \mathbb{E}[Y \mid X = x, \mathrm{do}(A = 0)].

Given an observational dataset :math:`(X_i, A_i, Y_i)_{i=1}^n`, it is a
challenging problem to estimate heterogeneous effects. WhyNot allows
benchmarking of individual treatment effect estimations by returning indivudal
level counterfactuals, i.e. both :math:`Y_{i, \mathrm{do}(A=0)}` and
:math:`Y_{i, \mathrm{do}(A=1)}` for each sample :math:`i`.

**Case-study: Opioid Epidemic Simulator**
To illustrate this, we consider the same study using the opioid epidemic
simulator presented in the section on :ref:`average-treatment-effects`.

.. code:: python

    import whynot as wn
    import numpy as np

    overdose_experiment = wn.opioid.Confounding

    dataset = overdose_experiment.run(num_samples=500)

    # True effects is a n x 1 vector of individual
    # level contrasts Y_i(1) - Y_i(0)
    dataset.true_effects

Estimating Heterogeneous Treatment Effects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
WhyNot provides a collection of estimators in the :func:`~whynot.causal_suite` to
estimate heterogeneous treatment effects. See :ref:`hte-estimators` for a
detailed list. Each estimator returns an :class:`~whynot.InferenceResult` with
the property ``individual_effects``. The code below shows how to use the
`causal forest estimator <http://arxiv.org/abs/1510.04342>`_ to estimate
individual treatment effects for the ``OverdoseConfounding`` experiment in the
previous section.

.. code:: python

    import whynot_estimators

    experiment  = wn.opioid.Confounding

    dataset = experiment.run(num_samples=100)

    # Estimate CATE using a causal forest
    estimate = whynot_estimators.causal_forest(
        dataset.covariates, dataset.treatment, dataset.outcome)

    # Compute MSE for HTE estimates
    mse = np.mean((estimate.individual_effects - dataset.true_effects) ** 2)



.. _Pearl 2009: https://dl.acm.org/citation.cfm?id=1642718
.. _Imbens and Rubin 2015: https://dl.acm.org/citation.cfm?id=2764565

