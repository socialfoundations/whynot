Usage
=====

.. toctree::
   :titlesonly:

   causal_inference
   causal_graph_discovery
   reinforcement_learning

.. rubric:: Causal Inference

WhyNot provides a powerful framework that generates data to stress-test methods
for estimating both :ref:`average-treatment-effects` and
:ref:`heterogeneous-treatment-effects` in a wide variety of simulated
environments.  After generating data with WhyNot, the package provides a
collection of :ref:`causal estimators <estimators>` to get started estimating
treatment effects, though users are also encouraged to try out their own
estimators.

All of the :ref:`simulators` implemented in WhyNot come equipped with a number
of causal inference experiments. Beyond these experiments, WhyNot easily and
flexibly supports the creation of new experiment designs to probe other causal
questins.  :ref:`designing-new-experiments` describes how to use primitives in
WhyNot to construct new experiments.


.. rubric:: Causal Graph Discovery

WhyNot also provides tools to automatically construct causal graphs associated
with runs of the simulators and to generate causal graphs for experiments
implemented in WhyNot. Equipped with these graphs, users can go beyond causal
inference and study problems of causal structure discovery.  See
:ref:`causal-graph-discovery` for more details on how to automatically construct
causal graphs in WhyNot and how to use these graphs to probe questions in causal
discovery.


.. rubric:: Sequential Decision and Reinforcement Learning

WhyNot is also an excellent test bed for sequential decision making and
reinforcement learning in diverse dynamic environments. WhyNot offers RL
environments compatible with the OpenAI Gym API style, so that existing code for
OpenAI Gym can be adapted for WhyNot with minimal changes. See
:ref:`reinforcement-learning` for more details on how to use available
environments in WhyNot and how to define new custom environments on top of
WhyNot simulators.
