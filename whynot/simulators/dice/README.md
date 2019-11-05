# DICE
The [Dynamice Integrated Climate Economy Model
(DICE)](https://sites.google.com/site/williamdnordhaus/dice-rice) is a
computer-based integrated assessment model developed by 2018 Nobel Laureate that
“integrates in an end-to-end fashion the economics, carbon cycle, climate
science, and impacts in a highly aggregated model that allows a weighing of the
costs and benefits of taking steps to slow greenhouse warming.”

The DICE model has a set of 26 state and 54 simulation parameters to
parameterize the dynamics. We omit listing all of them here are refer the reader
to the API documentation (DICE) for more details.


### State Variables
For a complete list of states, see the documentation or the
[code](simulator.py). Examples of state variables include

* Carbon concentration in atmosphere
* CO2-equivalent emissions
* Temperature of atmosphere
* Per capita consumption
* Investment
* Output net environmental damages

### Simulation Parameters
For a complete list of parameters, see the documentation or the
(code)[simulator.py]. Examples of simulation parameters include

* Initial world population 
* Growth rate
* Depreciation rate on capital
* Industion emissions 
* Initial base carbon price 
* Fraction of emissions under control in 2010
