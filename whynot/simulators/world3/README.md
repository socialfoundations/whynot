# World3
[World3](https://en.wikipedia.org/wiki/World3) is a systems dynamics model
developed by the Club of Rome in the early 1970s to illustrate the interactions
between population growth, industrial development, and the limitations of the
natural environment over time.  The model is a differential equation model with
13 state variables and 245 algebraic equations governing their evolution over
time.

### State Variables
* Population age 0 to 14
* Population age 15 to 44
* Population age 45 to 64
* Population age 65 and over
* Industrial capital
* Service capital
* Arable land
* Potentially arable land
* Urban industrial land
* Land fertility
* Nonrenewable resources
* Persistent pollution

### Simulation Parameters
* Policy year (year of intervention)
* Industrial capital output ratio
* Average lifetime of industrial capital
* Fraction of industrial output allocated to consumption
* Average lifetime of service capital
* Service capital output ratio
* Land yield factor
* Nonrenewable resource usage factor
* Persistent pollution generation factor

There are many more simulation parameters in World3 than those listed here.
The parameters enumerated above are all of the scalar parameters, and we omit
parameters corresponding to tabular functions. We only manipulate this restrict
set for simplicity.

## Experiments Implemented
1. Pollution generation reduction 
    - Reduce the rate of pollution generation from 1975 onward by 15%
    - The measure outcome is the total world population in 2050.
    - Experiments:
        1. **RCT**: Treatment is assigned randomly with probability p (a user-specified parameter)
        2. **Observed confounding**: Rollouts in the top decile of persistent
        pollution in 1975 receive treatment with probability p, and the
        remaining rollouts receive treatment with probability 1-p. All state
        variables in 1975 are observed to block backdoor paths.
        3. **Unobserved confounding**: Treatment assignment is the same as
        observed confounding. Persistent pollution in 1975 is always observed. A
        user-specified subset of the remaining state variables is unobserved,
        however, which generates unobserved confounding.
        4. **Mediation**: Treatment assignment is the same as observed
        confounding. All state variables are observed in 1975 to block backdoor
        paths. A user-specified subset of state variables in a year > 1975 is
        also observed to generate problems with mediation.

2. Natural resource usage reduction
    - Reduce the rate of natural resource usage from 1975 onward by 50%.
    - The outcome is the total world population in 2050.
    - Experiments: Similar to pollution generation.
        1. **RCT**: Treatment is randomly assigned with a user-specified
        probability p.
        2. **Observed confounding**: Rollouts in the bottom 20% of nonrenewable
        resources in 1975 receive treatment with probability p, and the
        remaining rollouts receive treatment with probability 1-p.
        3. **Unobserved confounding**: Treatment assignment the same as observed
        confounding. Nonrenewable resources in 1975 is always observed, and a
        user-specified subset of the remaining state variables is unobserved to
        generate unobserved confounding.
        4. **Mediation**: Include a subset of state variables from a year >1975.
