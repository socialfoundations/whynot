# World 2 
World 2 is a systems dynamics model developed by [Jay
Forrester](https://en.wikipedia.org/wiki/Jay_Wright_Forrester) to demonstrate
the tension between industrial growth and natural resource limitations.  The
model was used to study how limitations imposed natural resource constraints can
lead to slowdowns in industrial growth and eventual population collapse.

## Simulator
The model is a system of differential equations in 5 variables corresponding to
quantities and 43 algebraic equations governing their evolution over time.

### State Variables
* Population
* Natural resources
* Capital investment
* Pollution
* Fraction of capital investment in agriculture

### Simulation Parameters
* Policy year (year of intervention)
* Birth rate
* Death rate
* Effective capital investment ratio
* Natural resources usage
* Land area
* Population density
* Food coefficient
* Capital investment generation rate
* Capital investment discard rate
* Pollution rate


## Experiments
1. What is the effect of increasing capital investment in 1970 on world
population in 2000?
    - Treatment is increasing capital investment by 1% in 1970.
    - Measure the total world population in 2000
    - Experiments:
        1. **RCT**: Treatment is randomly assigned to each rollout with
        probability p.
        2. **Observed confounding**: States with the top 25\%
        of natural resources at time 0 are treated with probability p, and
        otherwise they are treated with probability 1-p. All of the state
        variables are observed.
        3. **Observed confounding without positivity**: States with the top 25\%
        of natural resources at time 0 are treated with probability 1. All of the state
        variables are observed.
        4. **Mediation**: Treatment assignment is the same as with observed
        confounding. Additional state variables from time > 1970 are observed,
        which gives rise to problems with mediation.
