.. _simulators:

Simulators
==========
WhyNot provides a large number of simulated environments from fields ranging
from economics to epidemiology. Each simulator comes equipped with a
representative set of causal inference experiments and will export a uniform
Python interface that makes it easy to construct new causal inference
experiments in these environments.

The simulators in WhyNot can be split into two main categories: system
dynamics models which use differential equations to model interactions between
different components of a system over time and agent-based models which
represent a system as a collection of interacting, heterogeneous agents. Below
we describe the set of simulators currently available in WhyNot.

*Disclaimer:* The simulators in our work are not intended to be realistic
models of the world, but rather realistic models of the technical difficulties
that the real world poses for causal inference. In many cases, the simulators
present in WhyNot have been disputed and criticized as models of the real
world.

Overview
--------
Dynamical systems based simulators:

* :ref:`dice-simulator`
* :ref:`adams-hiv-simulator`
* :ref:`world3-simulator`
* :ref:`world2-simulator`
* :ref:`opioid-simulator`
* :ref:`lotka-volterra-simulator`
* :ref:`lending-simulator`
* :ref:`repeated-classification`

Agent-based simulators:

* :ref:`incarceration-simulator`
* :ref:`civil-violence-simulator`
* :ref:`schelling-simulator`

Others:

* :ref:`lalonde-simulator`

Adding more simulators to WhyNot is easy and allows one to take advantage of
WhyNot tools for rapidly creating and running new causal inference experiments.
For more information, see :ref:`adding-a-simulator`. For additional API
reference, including specification of simulator states, configurations, and
supported interventions, see :ref:`Simulator API <simulator_api>`.


.. _dice-simulator:

Dynamic Integrated Climate Economy Model (DICE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`The Dynamice Integrated Climate Economy Model (DICE)
<https://en.wikipedia.org/wiki/DICE_model>`_ is a computer-based integrated
assessment model developed by 2018 Nobel Laureate William Nordhaus that
“integrates in an end-to-end fashion the economics, carbon cycle, climate
science, and impacts in a highly aggregated model that allows a weighing of the
costs and benefits of taking steps to slow greenhouse warming."

The DICE model has a set of 26 state and 54 simulation parameters to
parameterize the dynamics. We omit listing all of them here are refer the
reader to the API documentation (:ref:`dice`) for more details.

.. _adams-hiv-simulator:

Adams HIV (ODE-based HIV simulator)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Differential equation simulator of HIV based on

Adams, Brian Michael, et al.  *Dynamic multidrug therapies for HIV: Optimal and
STI control approaches.* North Carolina State University. Center for Research in Scientific Computation, 2004.  APA.

The Adams HIV model has a set of 6 state and 20 simulation parameters to
parameterize the dynamics. We omit listing all of them here are refer the reader to the API documentation (:ref:`hiv`) for more details.

.. _world3-simulator:

World3
^^^^^^
`World3 <https://en.wikipedia.org/wiki/World3>`_ is a systems dynamics model
commisioned by the Club of Rome in the early 1970s to illustrate the interactions between population growth, industrial development, and the
limitations of the natural environment over time.

The model is a differential equation model with 13 state variables and 245
algebraic equations governing their evolution over time.

**State Variables**

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

**Simulation Parameters**

* Policy year (year of intervention)
* Industrial capital output ratio
* Average lifetime of industrial capital
* Fraction of industrial output allocated to consumption
* Average lifetime of service capital
* Service capital output ratio
* Land yield factor
* Nonrenewable resource usage factor
* Persistent pollution generation factor

While there are many more simulation parameters in World3 than those listed
here, the parameters enumerated above are all of the scalar parameters, For
brevity and clarity's sake, we have omitted parameters corresponding to
tabular functions.

.. _world2-simulator:

World2
^^^^^^
World 2 is a systems dynamics model developed by `Jay Forrester
<https://en.wikipedia.org/wiki/Jay_Wright_Forrester>`_ to demonstrate the
tension between industrial growth and natural resource limitations. The model
is a precursor to the World3 model and, although it was used to study similar
questions, it represents different dynamics.

The model is a system of differential equations in 5 variables corresponding to
quantities and 43 algebraic equations governing their evolution over time.

**State Variables**

* Population
* Natural resources
* Capital investment
* Pollution
* Fraction of capital investment in agriculture

**Simulation Parameters**

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

.. _opioid-simulator:

Opioid Epidemic Simulator
^^^^^^^^^^^^^^^^^^^^^^^^^
The opioid epidemic simulator is a system dynamics model of the US opioid
epidemic developed by `Chen et al.
<https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2723405>`_ (JAMA,
2019). The model is calibrated based on past opioid use data from the Center
for Disease Control and was developed to simulate the effect of interventions
like reducing the number of new non-medical users of opioids on future opioid
overdose deaths in the United States. The simulator is a time-varying
differential equations model in 3 variables. For a complete description,
please refer to the appendix of `Chen et al.
<https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2723405>`_.

**State Variables**

* Number of people with non-medical use of prescription opioids
* Number of people with prescription opioid use disorder (OUD)
* Number of people with illicit opioid use

**Simulation Parameters**

* Annual incidence of

  * Non-medical prescription opioid use
  * Incidence of illicit opioid use
* Annual overdose mortality rate for

  * Non-medical prescription opioid use
  * OUD
  * Illicit opioid use
* Annual transition rate

  * From non-medical prescription opioid use to OUD
  * From non-medical prescription opioid use to illicit opioid use
  * From OUD to illicit opioid use
* Annual exit rate (either stop using opioids or die from non-opioid causes) for

  * Non-medical opioid use
  * OUD
  * Illicit opioid use


.. _civil-violence-simulator:

Civil Violence Simulator
^^^^^^^^^^^^^^^^^^^^^^^^
Civil Violence is an agent-based model of civil violence `introduced by Joshua
Epstein in 2002 <http://www.pnas.org/content/99/suppl_3/7243>`_. The model was
originally used to study the complex dynamics of decentralized rebellion and
revolution and to examine the state's efforts to counter these dynamics. The
model consists of two types of actors: agents and cops. Agents are
heterogenous, and their varied features make them more or less likely to
actively rebel against the state. The rich dynamics of the model emerge from
the interaction between agents and between agents and cops: agents are more
likely to begin rebel if other agents start to rebel, and the cops attempt to
arrest rebelling agents.

**Agents**
The agent-based simulator contains both agents and cops. Cops are homogenous,
while agents are individually endowed with the following (parameterized) qualities:

* Experienced hardship
* Belief in regime legitimacy
* Vision- number of adjacent squares an agent can inspect
* Rebellion threshold
* Risk aversion

**Simulation Parameters**

* Grid size (height and width)
* Density of cops
* Density of agents
* Cop vision- how many adjacent squares cops can inspect
* Maximum jail term length
* Prison interaction term
* Arrest probability constant (for calibration)

The implementation of this simulator is taken from the `examples <https://github.com/projectmesa/mesa/tree/master/examples>`_ of the `mesa library <https://github.com/projectmesa>`_.

.. _incarceration-simulator:

Incarceration Simulator
^^^^^^^^^^^^^^^^^^^^^^^
The incarceration simulator is based on the paper:

    Lum K, Swarup S, Eubank S, Hawdon J. *The contagious nature of
    imprisonment: an agent-based model to explain racial disparities in
    incarceration rates*.
    J R Soc Interface. 2014;11(98):20140409. `doi:10.1098/rsif.2014.0409
    <https://dx.doi.org/10.1098%2Frsif.2014.0409>`_

The paper proposes an agent-based model that models incarceration as
"contagious" in the sense that social ties to incarcerated individuals lead to
a higher risk of being imprisoned. The simulation occurs on a fixed set of
agents with a fixed set of social ties. What varies is the randomness with
which incarceration is passed on and randomness in sentence length. Transition
probabilities, and the sentence length distribution are based on real data.
The paper shows that higher-on-average sentence lengths for black individuals
than for whites lead to a disparity in incarceration rates that resembles the
one observed in the United States.


.. _lotka-volterra-simulator:

Lotka-Volterra Model
^^^^^^^^^^^^^^^^^^^^
Lotka-Volterra is a classical differential equation model of the interactions
between predator and prey in a single ecosystem. It serves as a simple example
to showcase how to use WhyNot to construct causal inference problems from
dynamical systems.  The model was originally developed to understand and
explain perplexing fishery statistics during World War I- namely why the
hiatus of fishing during the war led to an observed increase in the number of
predators.

For more details, see `Scholl 2012
<https://pdfs.semanticscholar.org/f314/7c9d2e43aafc492852f552990a3b21315ca5.pdf?_ga=2.132703694.1945084113.1556061073-1443175395.1541897531>`_.

The simulator is system of ordinary differential equations in two variables.
For a complete description, see
`here <https://scipy-cookbook.readthedocs.io/items/LoktaVolterraTutorial.html>`_.

**State Variables**

* Number of foxes
* Number of rabbits

**Simulation Parameters**

* Policy year (year of intervention)
* Rabbit growth factor
* Rabbit death factor
* Fox death factor
* Fox growth factor

.. _lending-simulator:

Lending Simulator
^^^^^^^^^^^^^^^^^
The Lending simulator is based on the paper:

    Liu, L., Dean, S., Rolf, E., Simchowitz, M., & Hardt, M. (2018, July).
    Delayed Impact of Fair Machine Learning. In International Conference on
    Machine Learning (pp. 3156-3164). Chicago.

The paper proposes a simple lending model in which individuals apply for
loans, a lending institution approves or denies the loan on the basis of the
individual's credit score, and subsequent loan repayment or default in turn
changes the individual's credit score. Credit scores and repayment probabilities
are based on real FICO data. In this dynamic setting, the paper shows that
static fairness criterion do not in genearl promote improvement over time and
can indeed cause active harm.

.. _repeated-classification:

Repeated Classification Simulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Repeated Classification simulator is based on the paper:

    Hashimoto, T., Srivastava, M., Namkoong, H., & Liang, P. (2018, July). Fairness
    Without Demographics in Repeated Loss Minimization. In International Conference
    on Machine Learning (pp. 1929-1938).

The paper proposes a simplified model of interaction between individuals from
different subgroups and standard machine learning classifiers based on empirical
risk minimization. In the model, decreases in accuracy on different subgroups
cause individuals to exit the system, further decreasing accuracy in these
subgroups and creating a negative feedback loop. The paper shows that, when
combined with repeated empirical risk minimization, even initially fair models
can become unfair over time if this dynamic is not accounted for.


.. _schelling-simulator:

Schelling Model
^^^^^^^^^^^^^^^
The `Schelling model
<https://www.stat.berkeley.edu/~aldous/157/Papers/Schelling_Seg_Models.pdf>`_
is a classic agent-based model originally used to illustrate how weak
individual preferences regarding one's neighbhors can lead to global
segregation of entire cities. In the model, individuals prefer to live where
at least some fraction of their neighbors are the same race as they are and
will move if this constraint is not met. As this process is iterated, an
originally well-mixed city rapidly becomes segregated by group.

**Agents**
The agents in Schelling's model are labeled either Type 0 or Type 1,
corresponding to members of the majority or minority class.

**Simulation Parameters**

* Grid size (height, width)
* Agent density
* Percentage of minority agents
* Homophily
* Education boost (how much receiving ``education`` decreases homophily)
* Percentage of agents receiving education

The implementation of this simulator is taken from the `examples <https://github.com/projectmesa/mesa/tree/master/examples>`_ of the `mesa library <https://github.com/projectmesa>`_.

.. _lalonde-simulator:

LaLonde Synthetic Outcome Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Lalone simulator is based on data from `Robert LaLonde's 1986 study
<https://www.jstor.org/stable/1806062>`_ evaluating the impact of the National
Supported Work Demonstration, a labor training program, on post-intervention
income levels. Since the actual function mapping the measured covariates to
the observed outcomes is unknown, we instead simulate random functions of
varying complexity on the data to generate synthetic outputs. This procedure
allows us to generate causal inference problems with response surfaces of
varying, but known complexity.

In the Lalonde data, the function mapping covariates :math:`X` to outcome
:math:`Y` is unknown, and it is impossible to simulate ground truth. Therefore,
following `Hill et al. <https://arxiv.org/abs/1707.02641>`_, we replace the
true outcome :math:`Y` with one generated by functions :math:`f_0, f_1`,
corresponds to control and treatment, as follows. Let :math:`W` denote
treatment assignment.
Then,

.. math::
    f_0(X) = Y(0),
    f_1(X) = Y(1),
    Y = Y(W).


