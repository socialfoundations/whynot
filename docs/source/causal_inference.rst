Causal Inference
================
Estimating treatment effects is a core problem in causal inference. WhyNot
provides a powerful framework that generates data to stress-test methods for
estimating both :ref:`average-treatment-effects` and
:ref:`heterogeneous-treatment-effects` in a wide variety of simulated
environments.

All of the :ref:`simulators` implemented in WhyNot come equipped with
experiments in both of these settings. Moreover, WhyNot provides a clean
framework to allow the user to implement new experiments to explore issues not
addressed by the fixed set of benchmarks. See :ref:`designing-new-experiments`
for a detailed discussed of framework and API for creating new benchmarks.

.. toctree::
    :titlesonly:
    
    treatment_effects 
    estimators
    designing_experiments
