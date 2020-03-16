![WhyNot Logo](docs/source/_static/WhyNot_fullcolor.svg)

[![Build Status](https://travis-ci.com/zykls/whynot.svg?token=ERpRX6SmHRsKJ8dNb4QV&branch=master)](https://travis-ci.com/zykls/whynot)
[![Documentation Status](https://readthedocs.com/projects/whynot-docs/badge/?version=latest)](https://whynot-docs.readthedocs-hosted.com/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**WhyNot** is a Python package that provides an experimental sandbox for
decisions in dynamics, connecting tools from causal inference and reinforcement
learning with challenging dynamic environments.  The package facilitates
developing, testing, benchmarking, and teaching causal inference and sequential
decision making tools.

For more detailed information, check out the [documentation](https://whynot-docs.readthedocs-hosted.com/en/latest/).

## Table of Contents
1. [Basic installation instructions](#basic-installation-instructions)
2. [Quick start examples](#quick-start-examples)
    - [Causal inference](#causal-inference)
    - [Sequential decision making](#sequential-decision-making)
    - [Strategic classification](#strategic-classification)
3. [Simulators in WhyNot](#simulators-in-whynot)
4. [Using estimators in R](#using-estimators-in-r)
5. [Frequently asked questions](#frequently-asked-questions)

WhyNot is still under active development! If you find bugs or have feature
requests, please file a 
[Github issue](https://github.com/zykls/whynot/issues). We welcome all kinds of issues, especially those related to correctness, documentation, performance, and new features.

## Basic installation instructions
1. (Optionally) create a virtual environment
```
python3 -m venv whynot-env
source whynot-env/bin/activate
```
2. Install via pip
```
pip install whynot
```

You can also install WhyNot directly from source.
```
git clone https://github.com/zykls/whynot.git
cd whynot
pip install -r requirements.txt
```

## Quick start examples

### Causal inference
Every simulator in WhyNot comes equipped with a set of experiments probing
different aspects of causal inference. In this section, we show how to run
experiments probing average treatment effect estimation on the [World3
simulator](https://whynot-docs.readthedocs-hosted.com/en/latest/simulators.html#world3-simulator). 
World3 is a dynamical systems model that studies the interplay between natural
resource constraints, population growth, and industrial development.

First, we examine all of the experiments available for World3.
```py
import whynot as wn
experiments = wn.world3.get_experiments()
print([experiment.name for experiment in experiments])
#['world3_rct', 'world3_pollution_confounding', 'world3_pollution_unobserved_confounding', 'world3_pollution_mediation']
```
These experiments generate datasets both in the setting of a pure randomized
control trial (`world3_rct`), as well as with (unobserved) confounding and
mediation. We will run a randomized control experiment. The description property
offers specific details about the experiment.
```py
rct = wn.world3.PollutionRCT
rct.description
#'Study effect of intervening in 1975 to decrease pollution generation on total population in 2050.'
```
We can run the experiment using the experiment `run` function and specifying a
desired sample size `num_samples`. The experiment then returns a causal
`Dataset` consisting of the covariates for each unit, the treatment assignment,
the outcome, and the ground truth causal effect for each unit. All of this data
is contained in numpy arrays, which makes it easy to connect to causal
estimators.
```py
import numpy as np

dset = rct.run(num_samples=500, show_progress=True)
(X, W, Y) = dset.covariates, dset.treatments, dset.outcomes
treatment_effect = np.mean(dset.true_effects)

# Plug-in your favorite causal estimator
estimated_ate = np.mean(Y[W == 1.]) -  np.mean(Y[W  == 0.])
```

WhyNot also enables you to run a large collection of causal estimators on the
data for benchmarking and comparison. The main function to do this is the
`causal_suite` which, given the causal dataset, runs all of the estimators on
the dataset and returns an `InferenceResult` for each estimator containing its
estimated treatment effects and uncertainty estimates like confidence intervals. 

```py
# Run the suite of estimates
estimated_effects = wn.causal_suite(
    dataset.covariates, dataset.treatments, dataset.outcomes)

# Evaluate the relative error of the estimates
true_sate = dataset.sate
for estimator, estimate in estimated_effects.items():
    relative_error = np.abs((estimate.ate - true_sate) / true_sate)
    print("{}: {:.2f}".format(estimator, relative_error))
# ols: 0.50
# propensity_weighted_ols: 0.51
# propensity_score_matching: 0.28
# matching: 0.75
# causal_forest: 0.06
# tmle: 0.06
```

In addition to experiments studying average treatment effect, WhyNot also supports
causal inference experiments studying
1. Heterogeneous treatment effects,
2. Time-varying treatment policies
3. Causal structure discovery

### Sequential decision making
WhyNot supports experimentation with sequential decision making and
reinforcement learning via unified interface with the [OpenAI
gym](https://github.com/openai/gym). In this section, we give a simple example
showing how to use the [HIV simulator](https://whynot-docs.readthedocs-hosted.com/en/latest/simulators.html#adams-hiv-simulator)
for sequential decision making experiments.

First, we initialize the environment and set the random seed.
```py
import whynot.gym as gym

env = gym.make('HIV-v0')
env.seed(1)
```
Observations in the simulator are a set of 6 states, capturing infected and
uninfected T-lymphocytes, macrophages, immune response, and copies of free
virus. Actions correspond to choosing between different drugs and dosages for
treatment.


For illustration, we repeatedly chose actions, which correspond to treatment
policy decisions, in the environment and measure both the next state and the
reward. In this case, the reward weighs the strength of the immune response, the
virus count, and the cost of the chosen treatment.
```py
observation = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Replace with your treatment policy
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
```
For more details on the simulation, as well as a fully worked out policy
gradient example, see [this notebook](https://github.com/zykls/whynot/blob/master/examples/reinforcement_learning/hiv_simulator.ipynb).  


### Strategic classification
Beyond settings typically studied in sequential decision making, WhyNot also
supports experiments with standard supervised learning algorithms in dynamic
settings. In this section, we show how to use WhyNot to study the performance of
classifiers when individuals being classified [*behave strategically*] to
improve their outcomes, a problem sometimes called [strategic
classification](https://arxiv.org/abs/1506.06980).


First, we set up the [credit
environment](https://whynot-docs.readthedocs-hosted.com/en/latest/simulators.html#credit).
```py
import whynot.gym as gym

env = gym.make('Credit-v0')
env.seed(1)
```
Observations in this environment correspond to a dataset of features for each
individual and a label indicating whether they experience financial distress
from the Kaggle [GiveMeSomeCredit dataset](https://www.kaggle.com/c/GiveMeSomeCredit).
```py
dataset = env.reset()
```
Actions in the environment correspond to choosing a classifier to predict
default. In response, individuals then *strategically adapt* their features in
order to obtain a more favorable credit score. The subsequent observation is the
adapted features, and the reward is the classifier's loss on this distribution
```py
theta = env.action_space.sample() # Your classifier
dataset, loss, done, info = env.step(theta)
```
We can then experiment with the long-term equilibrium arising from repeatedly
updating the classifier to cope with strategic response.
```py
def learn_classifier(features, labels):
    # Replace with your learning algorithm
    return env.action_space.sample()

dataset = env.reset()
for _ in range(100):
    theta = learn_classifier(dataset["features"], dataset["labels"])
    dataset, loss, _ = env.step(theta)
```
For more details on the simulation and a complete example showing the
standard retraining procedures perform in a strategic setting, see [this
notebook](https://github.com/zykls/whynot/blob/master/examples/dynamic_decisions/performative_prediction.ipynb).

Beyond strategic classification, WhyNot also supports simulators and experiments
evaluating other aspects of machine learning, e.g. [fairness
criteria](https://github.com/zykls/whynot/blob/master/examples/dynamic_decisions/delayed_impact.ipynb),
in dynamic settings.

For more examples and demonstrations of how to design and conduct
experiments in each of these settings, check out 
[usage](https://whynot-docs.readthedocs-hosted.com/en/latest/usage.html) and
our collection of
[examples](https://whynot-docs.readthedocs-hosted.com/en/latest/examples.html).


## Simulators in WhyNot
WhyNot provides a large number of simulated environments from fields ranging
from economics to epidemiology. Each simulator comes equipped with a
representative set of causal inference experiments and exports a uniform Python
interface that makes it easy to construct new causal inference experiments in
these environments, as well as an [OpenAI gym](https://github.com/openai/gym)
interface to perform reinforcement learning experiments in new environments.

The simulators in WhyNot currently include:
- [Adams HIV (ODE-based HIV simulator)](https://www.ncbi.nlm.nih.gov/pubmed/20369969)
- [Dynamic Integrated Climate Economy Model (DICE)](https://en.wikipedia.org/wiki/DICE_model)
- [World3](https://en.wikipedia.org/wiki/World3)
- [World2](https://scholar.google.com/scholar_lookup?title=World%20Dynamics&publication_year=1971&author=Jay%20W.%20Forrester)
- [Opioid Epidemic Simulator](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2723405)
- [Zika Control Simulator](https://www.sciencedirect.com/science/article/pii/S2211692316301084#b25)
- [Lotka-Volterra Model](https://en.wikipedia.org/wiki/Lotka–Volterra_equations)
- [Incarceration Simulator](https://royalsocietypublishing.org/doi/full/10.1098/rsif.2014.0409)
- [Civil Violence Simulator](https://www.pnas.org/content/99/suppl_3/7243)
- [Schelling Model](https://www.stat.berkeley.edu/~aldous/157/Papers/Schelling_Seg_Models.pdf)
- [LaLonde Synthetic Outcome Model](http://sekhon.berkeley.edu/matching/lalonde.html)
- [Delayed Impact](https://arxiv.org/abs/1803.04383)
- [Performative Prediction Credit Simulator](https://arxiv.org/abs/2002.06673)

For a detailed overview of these simulators, please see the [simulator
documentation](https://whynot-docs.readthedocs-hosted.com/en/latest/simulators.html).

## Using causal estimators in R
WhyNot ships with a small set of causal estimators written in pure Python.
To access other estimators, please install the companion library
[whynot_estimators](https://github.com/zykls/whynot_estimators), which includes a host of
state-of-the-art causal inference methods implemented in R.

To get the basic framework, run
```
pip install whynot_estimators
```
If you have R installed, you can install the `causal_forest` estimator by using
```
python -m  whynot_estimators install causal_forest
```
To see all of the available estimators, run
```
python -m  whynot_estimators show_all
```
See [whynot_estimators](https://github.com/zykls/whynot_estimators) for
instructions on installing specific estimators, especially if you do not have an
existing R build.


## Frequently asked questions
**1. Why is it called WhyNot?**

Why not?

**2. What are the intended use cases?**

WhyNot supports multiple use cases, some technical, some pedagogical, each
suited for a different group of users. We envision at least five primary use
cases:

- **Developing**: Researchers can use WhyNot in the process of developing new
  methods for causal inference and decision making in dynamic settings.  WhyNot
  can serve as a substitute for ad-hoc synthetic data where needed, providing a
  greater set of challenging test cases. 

- **Testing**: Researchers can use WhyNot to design robustness checks for
  methods and gain insight into the failure cases of these methods.

- **Benchmarking**: Practitioners can use WhyNot to compare multiple methods on
  the same set of tasks. WhyNot does not dictate any particular benchmark, but
  rather supports the community in creating useful benchmarks.

- **Learning**: Students of causality and dynamic decision making might find
  WhyNot to be a helpful training resource. WhyNot is easy-to-use and does not
  require much prior experience to get started with.

- **Teaching**: Instructors can use WhyNot as a tool students engage with to
  learn and solve problems.

**3. What uses are *not* intended?**

- **Basis of real-world policy and interventions**: The simulators included in
  WhyNot were selected because they offer realistic technical challenges for
  causal inference and dynamic decision making tools, not because they offer
  faithful models of the real world. In many cases, they have been contested or
  criticized as representations of the real world. For this reason, the
  simulators should not directly be used to design real-world interventions or
  policy.

- **Substitute for healthy debate**: Success in simulated environments does not
  guarantee success in real scenarios, but a failure in simulated environments
  can nonetheless lead to insight into weaknesses of a particular approach.
  WhyNot does not obviate the need for debate around common assumptions in
  causal inference.

- **Substitute for real world experiments and data**: WhyNot does not substitute
  for high-quality empirical work on real data sets. WhyNot is a tool for
  understanding and evaluating methods for causal inference and decision making
  in dynamics, not certifying their validity in real-world scenarios. 

- **Substitute for theory**: WhyNot can help create understanding in contexts
  where theoretical analysis is challenging, but does not reduce the need for
  theoretical guarantees and formal analysis.


**4. Why start from dynamical systems?**

Dynamical systems provide a natural setting to study causal inference. The
physical world is a dynamical system, and causal inference inevitably has to
grapple with data generated from some dynamical process. Moreover, the temporal
structure of the dynamics gives rise to nontrivial problem instances with both
confounding and mediation. Dynamics also naturally lead to time-varying causal
effects and allow for time-varying treatments and sequential decision making.

**5. What what simulators are included and why?**

WhyNot contains a range of different simulators, and an overview is provided in
the documentation
[here](https://whynot-docs.readthedocs-hosted.com/en/latest/).


**6. What’s the difference between WhyNot and CauseMe?**

[CauseMe](https://causeme.uv.es) is an online platform for benchmarking causal
discovery methods. Users can register and evaluate causal discovery methods on
an existing repository of data sets, or contribute their own data sets with
known ground truth. CauseMe is an excellent platform that we recommend in
addition to WhyNot. We encourage users to export data sets derived from WhyNot
and make them accessible through CauseMe. In this case, we ask that you
reference WhyNot.


**7. What’s the difference between WhyNot and CausalML?**

CausalML is a Python package that provides a range of causal inference methods.
The estimators provided by CausalML are available in WhyNot via the
[whynot_estimators](https://github.com/zykls/whynot_estimators)
package.  While WhyNot provides simulators and derived
experimental designs on synthetic data, CausalML focuses on providing
estimators. We made these estimators available for use on top of WhyNot.

**8. What’s the difference between WhyNot and EconML?**

EconML is a Python package that provides tools from machine learning and
econometrics for causal inference. Like CausalML, EconML focuses on providing
estimators, and we made these estimators available for use on top of WhyNot.

**9. How can I best contribute to WhyNot?**

Thanks so much for considering to contribute to WhyNot. The package is open
source and MIT licensed. We invite contributions broadly in a number of areas,
including the addition of simulators, causal estimators, sequential decision
making algorithms, documentation, performance improvements, code quality and
tests.
