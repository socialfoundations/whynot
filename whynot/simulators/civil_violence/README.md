# Civil Violence
Civil Violence is an agent-based model of civil violence [introduced by Joshua
Epstein in 2002.](http://www.pnas.org/content/99/suppl_3/7243) The model was
originally used to study the complex dynamics of decentralized rebellion and
revolution and the state's efforts to counter these dynamics. The model consists
of two types of actors: agents and cops. Agents are heterogenous, and their
varied features make them more or less likely to actively rebel against the
state. The rich dynamics of the model emerge from the interaction between agents
and between agents and cops: agents are more likely to begin rebel if other
agents start to rebel, and the cops attempt to arrest rebelling agents.

## Simulator

### Agents
The agent-based simulator contains both agents and cops. Cops are homogenous,
while agents are endowed with the following (parameterized) qualities:
* Experienced hardship
* Belief in regime legitimacy
* Vision- number of adjacent squares an agent can inspect
* Rebellion threshold
* Risk aversion

### Simulation Parameters
The simulation is governed with the following metaparamters
* Grid size (height and width)
* Density of cops
* Density of agents
* Cop vision- how many adjacent squares cops can inspect
* Maximum jail term length
* Prison interaction term
* Arrest probability constant (for calibration)

### Ground truth
Defining causal effects with unit interaction and interfence is subtle.
In all experiments, we consider the *unit-level causal effect*, which is the
marginal effect for a single unit when *no other agents are treated*.
We measure the ground-truth unit level causal effect by running a simulation in
which only one agent receives treatment, and then measuring the outcome.
Notice the discrepancy between the unit-level effect and the effect that would
be measured by treating several agents simultaneously, for instance, in an RCT.

## Experiments
1. What is the effect of increasing risk aversion on an individual's frequency
of rebellion?
    - Treatment raises risk aversion from 0.1 to 0.9.
    - Outcome is number of days an individual spends in *active rebellion*.
    - Perfect RCT---each unit is treated with probability 0.5.
    - Vary the strength of interaction by varying agent density. Since the
      number of agents is fixed, this corresponds to varying the size of the
      grid. 
