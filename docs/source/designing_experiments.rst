.. _designing-new-experiments:

Creating Experiments
====================
WhyNot easily and flexibly supports the creation of new causal experiments and
new experiment designs. Below, we describe how to use primitives in WhyNot to
construct a new experiment for both dynamical system models. For more examples
or inspiration for experiment creation and design, look at the many examples in
:ref:`experiment-examples`.

In WhyNot, the class :class:`~whynot.framework.DynamicsExperiment` encapsulates
the logic necessary to define a causal inference experiment on a dynamical
system simulator. For any causal inference experiment, we always need to
specify:

- The treatment :math:`A`
- The observed outcome :math:`Y`
- The observed covariates :math:`X`.
- The treatment assignment rule

In a dynamical system simulation, we need to further specify:

- The dynamics (i.e. which simulator we use)
- The initial state distribution.

Once we specify these components, the
:class:`~whynot.framework.DynamicsExperiment` does the rest of the work to
efficiently run the simulations, assign treatment, and construct the
observational dataset. Before launching into the details of each of these
components, we first give an example on the :ref:`lotka-volterra-simulator`.

.. code:: python

    import whynot as wn

    def sample_initial_state(rng):
        """Sample an initial state, i.e. a population of rabbits and foxes."""
        rabbits = rng.randint(10, 100)
        foxes = rabbits * rng.uniform(0.1, 0.8)
        return wn.lotka_volterra.State(rabbits=rabbits, foxes=foxes)

    def observed_outcome(run):
        """Compute the minimum fox population in the 20 years before year 80."""
        return np.min([run[year].foxes for year in range(60, 80)])

    rct = wn.DynamicsExperiment(
        name="lotka_volterra_rct",
        description="A RCT to determine effect of reducing rabbits needed to sustain a fox.",
        simulator=wn.lotka_volterra,
        simulator_config=wn.lotka_volterra.Config(fox_growth=0.75),
        intervention=wn.lotka_volterra.Intervention(time=30, fox_growth=0.4),
        state_sampler=sample_initial_state,
        propensity_scorer=0.5,
        outcome_extractor=observed_outcome,
        covariate_builder=lambda run: run.initial_state.values())

Each of the arguments to :class:`~whynot.framework.DynamicsExperiment``
determines one aspect of the causal experiment.

- The ``simulator`` specifies that the experiment is run on the Lotka-Volterra
  simulator.
- The ``simulator_config`` sets the parameters of the simulator.
- The ``intervention`` specifies what intervention to perform for the treatment
  group. In this case, the intervention corresponds to reducing the
  ``fox_growth`` parameter in year ``30``. 
- The ``state_sampler`` generates samples from the specified
  initial state distribution. 
- The ``propensity_scorer`` determines the probability of treatment assignment
  for a particular unit. Setting the ``propensity_scorer`` to a constant ``0.5``
  randomly assigns treatment with probability ``0.5`` to each unit. 
- The ``outcome_extractor`` computes the observed outcome :math:`Y` from a run
  of the simulator.
- The ``covariate_builder`` extracts the observed covariates :math:`X` from a
  simulator run.

This code can then be executed to generate the observational dataset.

.. code:: python

    >>> dataset = rct.run(num_samples=200, parallelize=True)

We can also generate experiments with confounding if the ``propensity_scorer``
depends on the simulator state. 

.. code:: python
    
    import whynot as wn

    def confounded_propensity_scores(run):
        """Return confounded treatment assignment probability.
        Treatment increases fox population growth. Therefore, we're assume
        treatment is more likely for runs with low initial fox population.
        """
        if run.initial_state.foxes < 20:
            return 0.8
        return 0.2

    confounding_exp = wn.DynamicsExperiment(
        name="lotka_volterra_confounding",
        description=("Determine effect of reducing rabbits needed to sustain a "
                     "fox. Treament confounded by initial fox population."),
        simulator=wn.lotka_volterra,
        simulator_config=wn.lotka_volterra.Config(fox_growth=0.75),
        intervention=wn.lotka_volterra.Intervention(time=30, fox_growth=0.4),
        state_sampler=sample_initial_state,
        propensity_scorer=confounded_propensity_scores,
        outcome_extractor=observed_outcome,
        covariate_builder=lambda run: run.initial_state.values())


In the previous two examples, we hard-coded several parameters into the
experiment specification. For instance, we set the treatment probabilities in
the confounding example to ``0.8`` and ``0.2`` depending on the initial state.
However, we often want to run experiments for a set of parameters. For instance,
rather then consider a single propensity score setting, we could study the
performance of a family of estimators as the *strength* of the confounding
varied. In WhyNot, the ``@parameter`` decorator allows us to do precisely that.


.. code:: python

    import whynot as wn
    
    @wn.parameter(name="propensity", default=0.9, 
               description="Treatment prob for group with low fox counts.")
    def confounded_propensity_scores(run, propensity):
        """Return confounded treatment assignment probability.
        Treatment increases fox population growth. Therefore, we're assume
        treatment is more likely for runs with low initial fox population.
        """
        if run.initial_state.foxes < 20:
            return propensity
        return 1. - propensity

    confounding_exp = wn.DynamicsExperiment(
        name="lotka_volterra_confounding",
        description=("Determine effect of reducing rabbits needed to sustain a "
                     "fox. Treament confounded by initial fox population."),
        simulator=wn.lotka_volterra,
        simulator_config=wn.lotka_volterra.Config(fox_growth=0.75),
        intervention=wn.lotka_volterra.Intervention(time=30, fox_growth=0.4),
        state_sampler=sample_initial_state,
        propensity_scorer=confounded_propensity_scores,
        outcome_extractor=observed_outcome,
        covariate_builder=lambda run: run.initial_state.values())

When a method is decorated with ``@parameter``, the ``run`` method of the ``DynamicsExperiment``
allows the parameter to be passed in. This make it very easy to generate a
sequence of observational datasets with as the parameter varies.

.. code:: python
    
    datasets = []
    for propensity in [0.5, 0.7, 0.9, 0.95]:
        dataset = confounding_exp.run(num_samples=1000, propensity=propensity)
        datasets.append(dataset)


As the above examples suggest, :class:`~whynot.framework.DynamicsExperiment` is
very flexible. For all of the details of permissible specifications of the
``state_sampler``, ``propensity_scorer``, etc., please refer to the 
:class:`API <whynot.framework.DynamicsExperiment>`.
