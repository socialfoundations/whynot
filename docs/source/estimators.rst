.. _estimators:

Estimators
==========

WhyNot comes equipped with a large collection of estimators to enable rapid
comparisons of methods on new benchmarks and with new experimental designs.

We hope this both allows users of causal inference tools to choose methods that
perform well in situations of interest. We also hope that WhyNot will allow
producers of new causal inference methods to rapidly evaluate their algorithms 
against a rich set of baselines on common datasets.


.. _ate-estimators:

Average Treatment Effect Estimators
-----------------------------------

* Linear Regression
* Matching, using the package `Matching <http://sekhon.berkeley.edu/matching/>`_.
* IP Weighting, using the package `WeightIt  <https://github.com/ngreifer/WeightIt>`_.
* Causal Forests, using the package `GRF <https://github.com/grf-labs/grf>`_.
* Causal Bart, using the package `BartCause <https://github.com/vdorie/bartCause>`_.
* TMLE, using the package `TMLE <https://cran.r-project.org/package=tmle>`_.
* Double machine learning, using the package `econml <https://github.com/microsoft/EconML>`_.


.. _hte-estimators:

Heterogeneous Treatment Effect Estimators
-----------------------------------------

* Causal Forests, using the package `GRF <https://github.com/grf-labs/grf>`_.
* Causal Bart, using the package `BartCause <https://github.com/vdorie/bartCause>`_.
* Double machine learning, using the package `econml <https://github.com/microsoft/EconML>`_.

The above are the readily available estimators in the WhyNot package.  It is
also easy to add estimators to the framework, see :ref:`adding-estimators`.

