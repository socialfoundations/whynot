Usage
=====

.. toctree::
   :titlesonly:

   designing_experiments
   treatment_effects
   causal_graph_discovery
   reinforcement_learning

.. rubric:: Setting Up An Experiment

WhyNot easily and flexibly supports the creation of new causal experiments and
new experiment designs. :ref:`designing-new-experiments` describes how to use
primitives in WhyNot to construct new experiments.

.. rubric:: Treatment Effect Estimation

Estimating treatment effects is a core problem in causal inference. WhyNot
provides a powerful framework that generates data to stress-test methods for
estimating both :ref:`average-treatment-effects` and
:ref:`heterogeneous-treatment-effects` in a wide variety of simulated
environments. 
All of the :ref:`simulators` implemented in WhyNot come equipped with
experiments in both of these settings.

.. rubric:: Causal Graph Discovery

WhyNot also provides tools to automatically construct causal graphs associated
with runs of the simulators and to generate causal graphs for experiments
implemented in WhyNot. Equipped with these graphs, users can go beyond
treatment effect estimation and study problems of causal structure discovery.
See :ref:`causal-graph-discovery` for more details on how to automatically
construct causal graphs in WhyNot and how to use these graphs to probe
questions in causal discovery.


.. rubric:: Sequential Decision and Reinforcement Learning

WhyNot is also an excellent test bed for sequential decision making and
reinforcement learning in diverse dynamic environments. WhyNot offers RL
environments compatible with the OpenAI Gym API style, so that existing code for OpenAI Gym can be adapted for WhyNot with minimal changes. See
:ref:`reinforcement-learning` for more details on how to use available
environments in WhyNot and how to define new custom environments on top of
WhyNot simulators.
