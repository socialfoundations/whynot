# Zika
The [Zika simulator](https://www.sciencedirect.com/science/article/pii/S2211692316301084) 
is dynamical systems model of transmission and control of the Zika virus
disease. The simulator is based on:

Momoh, Abdulfatai A., and Armin Fügenschuh. "Optimal control of intervention
strategies and cost effectiveness analysis for a Zika virus model." Operations
Research for Health Care 18 (2018): 99-111.

The intended purpose of the model was to study the efficacy of various
strategies for controlling the spread of the Zika virus: the use of treated
bednets, the use of condoms, a medical treatment of infected persons, and the
use of indoor residual spray (IRS). The dynamics of the Zika model govern the
evolution of 9 state variables, and the simulator has four control variables
and 20 simulation parameters.

### State Variables
The state variables are:

* susceptible_humans:   Number of susceptible humans
* asymptomatic_humans:  Number of asymptomatic infected human
* symptomatic_humans:   Number of symptomatic infected humans
* recovered_humans:     Number of recovered humans
* susceptible_humans:   Number of susceptible mosquitoes
* exposed_mosquitos:    Number of exposed mosquitoes
* infectious_mosquites: Number of infectious mosquitoes
* human_population:     Total number of human
* mosquito_population:  Total number of mosquitoes

The initial state corresponds to a small community of 1000 people, with a small
initial symptomatic infected population, as well as a large mosquite population.

### Simulation Parameters
We omit listing the entire set of simulation parameters and refer the reader to
the [code](simulator.py) or the documentation for details. For illustrative
purposes, the parameters include:

* Natural human death rate
* Natural mosquito death rate
* Disease induced death rate
* Spontaneous individual recovery
* Per capital biting rate of mosquitos
* etc...
