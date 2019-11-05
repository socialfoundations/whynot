# HIV
The [HIV simulator](https://www.ncbi.nlm.nih.gov/pubmed/20369969) is dynamical
systems model of the interaction between the immune system with the human
immunodeficiency virus (HIV). The simulator is based on:

Adams, Brian Michael, et al. *Dynamic multidrug therapies for HIV: Optimal and
STI control approaches*. North Carolina State University. Center for Research in
Scientific Computation, 2004. APA.

The intended purpose of the model was to study the efficacy of therapeutic
strategies for HIV. The model includes two types of treatments: reverse
transcriptase (RT) inhibitors and protease inhibitors (PIs). RT inhibitors are
more effective on CD4+ T-lymphocytes (T1) cells, while protease inhibitors are
more effective on macrophages (T2) cells. The dynamics of the HIV model govern
the evolution of 6 state variables and are controlled by 20 simulation
parameters.

### State Variables
The state variables are:

* uninfected_T1: uninfected CD4+ T-lymphocytes (cells/ml)
* infected_T1: infected CD4+ T-lymphocytes (cells/ml)
* uninfected_T2: uninfected macrophages (cells/ml)
* infected_T2: infected macrophages (cells/ml)
* free_virus: free virus (copies/ml)
* immune_response: immune response CTL E (cells/ml)

The initial state corresponds to an early infection state, defined by Adams et
corresponding to 1) adding one virus particle per ml of blood plasma, and 2)
adding low levels of infected T-cells.

### Simulation Parameters
We omit listing the entire set of simulation parameters and refer the reader to
the [code](simulator.py) or the documentation for details. For illustrative
purposes, the parameters include:

* CD4+ T-lymphocyte production rate
* CD4+ T-lymphocyte death rate
* CD4+ T-lymphocyte infection rate
* Immune-induced clearance rates
* Natural virus death rates
* Reverse Transcriptase inhibitors
* Protease inhibitors
* etc...


## Experiments Implemented
1. HIVConfounding
    * Study effects of increased drug efficacy on infected macrophages (cells/ml).
    * Introduces confounding because patients with high immune response and free virus are more likely to be treated.

2. HIVEnv
    * An environment for learning optimal treatment policies using the HIV
      simulator, in the setting proposed by Adams, et al.
    * There are 4 possible actions:
        - Action 0: no drug, costs 0
        - Action 1: protease inhibitor only, costs 1800
        - Action 2: RT inhibitor only, costs 9800
        - Action 3: both RT inhibitor and protease inhibitor, costs 11600
    * The reward at each step is defined based on the current state and the
    * action. We follow the optimization objective introduced in the original
    * paper by Adams et. al. Intuitively, the reward penalizes higher levels of
virus and rewards a strong immune response.
