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
    simulator_configs/zika
    simulator_configs/civil_violence
    simulator_configs/incarceration
    simulator_configs/lotka_volterra
    simulator_configs/schelling
    simulator_configs/lalonde
    simulator_configs/delayed_impact
    simulator_configs/credit


Dynamics
--------
.. autoclass:: whynot.dynamics.Run
    :members: initial_state, __getitem__

.. autoclass:: whynot.dynamics.BaseState

.. autoclass:: whynot.dynamics.BaseConfig

.. autoclass:: whynot.dynamics.BaseIntervention

.. autoclass:: whynot.dynamics.DynamicsExperiment
    :members: __init__, get_parameters, run


Reinforcement learning
----------------------
.. autoclass:: whynot.gym.envs.ODEEnvBuilder
    :members: __init__, reset, seed, step


Framework
---------

.. autoclass:: whynot.framework.Dataset

.. autoclass:: whynot.framework.ExperimentParameter

.. autoclass:: whynot.framework.ParameterCollection


Estimators
-----------

.. autofunction:: whynot.causal_suite

.. autoclass:: whynot.framework.InferenceResult

