# Lotka-Volterra
Lotka-Volterra is a classical differential equation model of the interactions
between predator and prey in a single ecosystem. It serves as a simple example
to showcase how to use WhyNot to construct causal inference problems from
dynamical systems.
The model was originally developed to understand and explain perplexing fishery
statistics during World War I- namely why the hiatus of fishing during the war
led to an observed increase in the number of predators. For more details, see
[Scholl 2012](https://pdfs.semanticscholar.org/f314/7c9d2e43aafc492852f552990a3b21315ca5.pdf?_ga=2.132703694.1945084113.1556061073-1443175395.1541897531).

## Simulator
The simulator is system of ordinary differential equations in two variables.
For a complete description, see
[here](https://scipy-cookbook.readthedocs.io/items/LoktaVolterraTutorial.html).

### State Variables
* Number of foxes
* Number of rabbits

### Simulation Parameters
* Policy year (year of intervention)
* Rabbit growth factor
* Rabbit death factor
* Fox death factor
* Fox growth factor


## Experiments
1. How does introducing alternate food supplies impact the minimum observed fox
population?
    - Treatment is reducing the ``fox growth`` factor from 0.75 to 0.4. ``Fox
      growth`` is how many rabbits are required to support a single fox.
    - Measured outcome is the minimum observed fox population over the last 10
      years of the simulation.
    - Experiments:
        1. **RCT**: Treatment is assigned randomly with a user-specified
        probability p.
        2. **Observed confounding**: Rollouts where the number of foxes is
        initially low, e.g. below 20 at time 0, are treated with probability
        p, and otherwise they are treated with probability 1-p. Both foxes and
        rabbits are observed.
        3. **Unobserved confounding**: Treatment is the same as in the observed
        confounding case. Howver, the number of foxes is not observed, and hence
        is an unobserved confounder.
