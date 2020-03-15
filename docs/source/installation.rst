.. _installation:

Installation
============

Basic Installation Instructions
-------------------------------

WhyNot supports Python3 on both OS X and Linux systems. We recommend using `pip`
for installation and isolating the installation inside a `virtualenv`_.

1. (Optionally) create a virtual environment.

.. code:: bash

    python3 -m venv whynot-env
    source whynot-env/bin/activate

2. Install the package.

.. code:: bash

    pip install whynot



Installing Other Estimators
---------------------------
By default, WhyNot ships with a small set of causal estimators written in pure
Python. Because many state-of-the-art causal inference routines are implemented
in R or have complex dependencies, you may wish to install the companion
`WhyNot-Estimators <https://github.com/zykls/whynot_estimators>`_ 
package. This package makes a large collection of these estimators, for 
instance, `causal forest`_, available for use with WhyNot.

To install WhyNot-Estimators and use the R-estimators, you will need a working 
installation of `R`_.  We recommend using `Anaconda`_ to  install `R`_, which 
will simplify dependency management. If you already have a working `R`_ 
installation and do not wish to install `Anaconda`_, skip the first two steps.

1. (Optional) Install `Anaconda`_.

2. (Optional) Create an R environment

.. code:: bash

    conda create --name whynot r-base r-essentials

3. Install the package! 

.. code:: bash

    pip install whynot_estimators

This installs the basic framework. To see all of the available estimators, run

.. code:: bash

    python -m whynot_estimators show_all

To install specific estimators, run

.. code:: bash

    python -m whynot_estimators install ESTIMATOR_NAME

Here, ``ESTIMATOR_NAME`` is the name of the estimator you wish to install.
For example, to install the `causal forest`, 

.. code:: bash
    
    python -m whynot_estimators install causal_forest

To install all of the estimators, run

.. code:: bash

    python -m whynot_estimators install all


.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _causal forest: https://arxiv.org/abs/1510.04342
.. _R: https://www.r-project.org
.. _virtualenv: https://virtualenv.pypa.io/en/stable/
