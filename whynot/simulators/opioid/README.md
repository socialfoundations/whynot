# Opioid epidemic simulator

This opioid epidemic simulator is a system dynamics model of the US opioid
epidemic developed by [Chen et
al.](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2723405) (JAMA, 2019).  
The model is calibrated based on past opioid use data from the Center for Disease
Control and was developed to simulate the effect of interventions like reducing
the number of new non-medical users of opioids on future opioid overdose deaths
in the United States.

## Simulator
The simulator is a time-varying differential equations model in 3 variables.
For a complete description, please refer to the appendix of [Chen et
al.](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2723405).

### State Variables
* Number of people with non-medical use of prescription opioids 
* Number of people with prescription opioid use disorder (OUD)
* Number of people with illicit opioid use 

### Simulation Parameters
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


## Experiments
1. What is the effect of reducing the annual incidence of non-medical
prescription opioid use?
    - Decrease annual incidence of non-medical prescription opioid use by 11.3% starting in 2015.
    - Measure the total number of overdose deaths from 2016 to 2025.
    - Experiments:
        1. **RCT**: Randomly assign treatment with probability 0.5
        2. **Observed confounding**: Rollouts with higher numbers of illicit
        overdose deaths in 2015 are more likely to be selected for treatment.
        Concretely, rollouts in the top decile of illicit overdose deaths
        receive treatment with probability p, and the remaining rollouts receive
        treatment with probability 1-p, for p > 0.5. The entire state in 2015 is
        observed.
        3. **Unobserved confounding**: Treatment assignment is the same as with
        observed confounding. However, only illicit overdose deaths in 2015 is
        observed, so there is unobserved confounding.
        4. **Mediation**: Treatment assignment is the same as with observed
        confounding. Additional state variables from a year after 2015 are
        also observed in order to generate mediation problems.
