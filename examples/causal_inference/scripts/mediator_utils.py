"""Utility functions for mediation experiment."""
import matplotlib.pylab as plt
import numpy as np


def error_bar_plot(experiment_data, results, title="", ylabel=""):

    true_effect = experiment_data.true_effects.mean()
    estimators = list(results.keys())

    x = list(estimators)
    y = [results[estimator].ate for estimator in estimators]

    cis = [
        np.array(results[estimator].ci) - results[estimator].ate
        if results[estimator].ci is not None
        else [0, 0]
        for estimator in estimators
    ]
    err = [[abs(ci[0]) for ci in cis], [abs(ci[1]) for ci in cis]]

    plt.figure(figsize=(12, 5))
    (_, caps, _) = plt.errorbar(x, y, yerr=err, fmt="o", markersize=8, capsize=5)
    for cap in caps:
        cap.set_markeredgewidth(2)
    plt.plot(x, [true_effect] * len(x), label="True Effect")
    plt.legend(fontsize=12, loc="lower right")
    plt.ylabel(ylabel)
    plt.title(title)
