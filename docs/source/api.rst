API
==========

.. _simulator_api:

Simulators
-----------
.. toctree::
    :maxdepth: 1
    
    simulator_configs/dice
    simulator_configs/hiv
    simulator_configs/world3
    simulator_configs/world2
    simulator_configs/opioid
    simulator_configs/civil_violence
    simulator_configs/incarceration
    simulator_configs/lotka_volterra
    simulator_configs/schelling
    simulator_configs/lalonde
    simulator_configs/lending


Dynamics
--------
.. autoclass:: whynot.dynamics.Run
    :members: initial_state, __getitem__

.. autoclass:: whynot.dynamics.DynamicsExperiment
    :members: __init__, get_parameters, run


Framework
---------

.. autoclass:: whynot.framework.Dataset

.. autoclass:: whynot.framework.ExperimentParameter

.. autoclass:: whynot.framework.ParameterCollection


Estimators
-----------

.. autofunction:: whynot.causal_suite

.. autoclass:: whynot.framework.InferenceResult

