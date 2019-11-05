# Schelling
The [Schelling
model](https://www.stat.berkeley.edu/~aldous/157/Papers/Schelling_Seg_Models.pdf)
is a classic agent-based model originally used to illustrate how weak individual
preferences regarding one's neighbhors can lead to global segregation of entire
cities. Each individual prefers to live where at least some fraction of his
neighbors are the same race as he is and will move if this constraint is not
met. As this process is iterated, an originally well-mixed city rapidly becomes
segregated by group.

## Simulator

### Agents
The agents in Schelling's model are labeled either Type 0 or Type 1,
corresponding to members of the majority or minority class.

### Simulation Parameters
Each run is parameterized by
* Grid size (height, width)
* Agent density
* Percentage of minority agents
* Homophily
* Education boost (how much receiving ``education`` decreases homophily)
* Percentage of agents receiving education

## Experiments
1. What is the impact of educating a subset of the population on segregation?
    - Each unit is a rollout of the Schelling model.
    - Treatment is randomly selecting 30% of the agents in the model to receive
      ``education`` which reduces their homophily.
    - Measure the fraction of agents that are segregated when the simulation
      concludes.
    - **RCT**: Assign treatment randomly with probability 25% to runs of the
      Schelling model.
