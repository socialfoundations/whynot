import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import *
import ipywidgets as widgets

import whynot as wn
from whynot.dynamics import DynamicsExperiment
from whynot.simulators import opioid
from whynot.simulators.opioid.experiments import (
    sample_initial_states,
    opioid_intervention,
    overdose_deaths,
)
from whynot_estimators import causal_forest


from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Palatino"]})
rc("text", usetex=False)
paper_rc = {
    "lines.linewidth": 3.0,
    "lines.markersize": 4,
    "lines.markeredgewidth": 2,
    "errorbar.capsize": 2,
    "figure.figsize": (10, 6),
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.titlesize": 28,
    "axes.labelsize": 24,
}
sns.set_context("paper", font_scale=3.3, rc=paper_rc)
sns.set_style("whitegrid")


def generate_data(data_dir="heterogeneous_example_data", verbose=True):
    if data_dir[-1] != "/":
        data_dir += "/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_filename = data_dir + "heterogeneous_example_data.json"

    sample_sizes = [100, 200, 500, 1000, 2000]
    illicit_exits = np.arange(0.28, 0.36 + 1e-6, 0.04)
    nonmedical_incidence_deltas = np.arange(-0.12, -0.1 + 1e-6, 0.01)

    try:
        data = json.load(open(data_filename))
        print("DATA FILE FOUND")
    except:
        print("DATA FILE NOT FOUND")
        data = {}

    for n in sample_sizes:
        data[str(n)] = data.get(str(n), {})
        for illicit_exit in illicit_exits:
            key = "{:.2f}".format(illicit_exit)
            data[str(n)][key] = data.get(key, {})
            for delta in nonmedical_incidence_deltas:
                key_ = "{:.2f}".format(delta)
                data[str(n)][key][key_] = data[str(n)][key].get(key_, {})

    for n in sample_sizes:
        for i, illicit_exit in enumerate(illicit_exits):
            for j, delta in enumerate(nonmedical_incidence_deltas):
                one_run_data = data[str(n)]["{:.2f}".format(illicit_exit)][
                    "{:.2f}".format(delta)
                ]
                if "true_effects" in one_run_data:
                    continue
                if verbose:
                    sim_number = i * len(illicit_exits) + j + 1
                    total_num_sims = len(illicit_exits) * len(
                        nonmedical_incidence_deltas
                    )
                    if sim_number == 1:
                        print("\nSIMULATING: SAMPLE SIZE {}".format(n))
                    print(
                        "  Running simulation {}/{}".format(sim_number, total_num_sims)
                    )
                experiment = DynamicsExperiment(
                    name="opioid_rct",
                    description=(
                        "Randomized experiment reducing nonmedical incidence of "
                        "opioid use in 2015."
                    ),
                    simulator=opioid,
                    simulator_config=opioid.Config(illicit_exit=illicit_exit),
                    intervention=opioid_intervention,
                    state_sampler=sample_initial_states,
                    propensity_scorer=0.5,
                    outcome_extractor=overdose_deaths,
                    covariate_builder=lambda run: run.initial_state.values(),
                )
                d = experiment.run(num_samples=n, nonmedical_incidence_delta=delta)
                one_run_data["covariates"] = tuple(
                    tuple(x) for x in d.covariates.astype(float)
                )
                one_run_data["treatments"] = tuple(d.treatments.astype(float))
                one_run_data["outcomes"] = tuple(d.outcomes.astype(float))
                one_run_data["true_effects"] = tuple(d.true_effects.astype(float))
    json.dump(data, open(data_filename, "w"))

    for n in sample_sizes:
        for i, illicit_exit in enumerate(illicit_exits):
            for j, delta in enumerate(nonmedical_incidence_deltas):
                one_run_data = data[str(n)]["{:.2f}".format(illicit_exit)][
                    "{:.2f}".format(delta)
                ]
                if "estimated_effects" in one_run_data:
                    continue
                if verbose:
                    sim_number = i * len(illicit_exits) + j + 1
                    total_num_sims = len(illicit_exits) * len(
                        nonmedical_incidence_deltas
                    )
                    if sim_number == 1:
                        print("\nESTIMATING: SAMPLE SIZE {}".format(n))
                    print(
                        "  Running estimation {}/{}".format(sim_number, total_num_sims)
                    )
                estimate = causal_forest.estimate_treatment_effect(
                    np.array(one_run_data["covariates"]),
                    np.array(one_run_data["treatments"]),
                    np.array(one_run_data["outcomes"]),
                )
                one_run_data["estimated_effects"] = tuple(estimate.individual_effects)
    json.dump(data, open(data_filename, "w"))


def generate_error_data(data_dir="heterogeneous_example_data", verbose=True):
    if data_dir[-1] != "/":
        data_dir += "/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_filename = data_dir + "heterogeneous_example_error_data.json"

    sample_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]

    try:
        data = json.load(open(data_filename))
        print("DATA FILE FOUND")
    except:
        print("DATA FILE NOT FOUND")
        data = {}

    num_iters = 20

    for n in sample_sizes:
        data[str(n)] = data.get(str(n), {})
        for i in range(num_iters):
            data[str(n)][str(i)] = data[str(n)].get(str(i), {})

    experiment = wn.opioid.RCT

    for n in sample_sizes:
        for i in range(num_iters):
            one_run_data = data[str(n)][str(i)]
            if "estimated_effects" in one_run_data:
                continue
            if verbose:
                if i == 0:
                    print("\nGENERATING ERROR DATA: SAMPLE SIZE {}".format(n))
                print("  Running iteration {}/{}".format(i + 1, num_iters))
            d = experiment.run(num_samples=n)
            one_run_data["covariates"] = tuple(
                tuple(x) for x in d.covariates.astype(float)
            )
            one_run_data["treatments"] = tuple(d.treatments.astype(float))
            one_run_data["outcomes"] = tuple(d.outcomes.astype(float))
            one_run_data["true_effects"] = tuple(d.true_effects.astype(float))
            estimate = causal_forest.estimate_treatment_effect(
                d.covariates, d.treatments, d.outcomes
            )
            one_run_data["estimated_effects"] = tuple(estimate.individual_effects)
            json.dump(data, open(data_filename, "w"))


def effects_histogram(
    true_effects,
    estimated_effects=None,
    num_bins=100,
    title=None,
    x_range=None,
    y_max=None,
    show_avg=True,
):
    plt.figure(figsize=(18, 8))
    if title is None:
        if estimated_effects is None:
            title = "True Heterogeneous Effects"
        else:
            title = "True and Estimated Heterogeneous Effects"
    plt.title(title, pad=20)

    if estimated_effects is not None and x_range is None:
        x_range = np.percentile(
            np.concatenate((true_effects, estimated_effects)), (5, 95)
        )

    true_effects_mean = np.mean(true_effects)
    plt.axvline(x=true_effects_mean, color="C0", linestyle="--")
    plt.hist(
        true_effects,
        bins=num_bins,
        range=x_range,
        density=True,
        alpha=0.5,
        color="C0",
        label="True Effects",
    )
    if estimated_effects is not None:
        estimated_effects_mean = np.mean(estimated_effects)
        plt.axvline(x=estimated_effects_mean, color="C1", linestyle="--")
        plt.hist(
            estimated_effects,
            bins=num_bins,
            range=x_range,
            density=True,
            alpha=0.5,
            color="C1",
            label="Estimated Effects",
        )

    if y_max:
        plt.ylim(top=y_max * 1.05)

    if show_avg:
        plt.text(
            true_effects_mean,
            plt.ylim()[1] * 0.95,
            "  True ATE",
            color="C0",
            fontsize=18,
        )
        if estimated_effects is not None:
            plt.text(
                estimated_effects_mean,
                plt.ylim()[1] * 0.9,
                "  Estimated ATE",
                color="C1",
                fontsize=18,
            )

    plt.xlabel("Effect Size")
    plt.ylabel("Density")
    plt.legend(fontsize=18)
    plt.show()


def effects_histogram_slider(
    data_dir="heterogeneous_example_data", show_estimated_effects=False
):
    if data_dir[-1] != "/":
        data_dir += "/"
    data_filename = data_dir + "heterogeneous_example_data.json"
    data = json.load(open(data_filename))

    x_min, x_max = np.inf, -np.inf

    # get x bounds
    for n in data.keys():
        for illicit_exit in data[n].keys():
            for delta in data[n][illicit_exit].keys():
                one_run_data = data[n][illicit_exit][delta]
                x_min = min(x_min, np.percentile(one_run_data["true_effects"], 10))
                x_max = max(x_max, np.percentile(one_run_data["true_effects"], 90))
                if show_estimated_effects:
                    x_min = min(
                        x_min, np.percentile(one_run_data["estimated_effects"], 10)
                    )
                    x_max = max(
                        x_max, np.percentile(one_run_data["estimated_effects"], 90)
                    )

    x_range = (x_min, x_max)
    y_max = 0
    num_bins = 100

    # get y_max of plot
    for n in data.keys():
        for illicit_exit in data[n].keys():
            for delta in data[n][illicit_exit].keys():
                one_run_data = data[n][illicit_exit][delta]
                weights, _ = np.histogram(
                    one_run_data["true_effects"],
                    bins=num_bins,
                    range=x_range,
                    density=True,
                )
                y_max = max(y_max, max(weights))
                if show_estimated_effects:
                    weights, _ = np.histogram(
                        one_run_data["estimated_effects"],
                        bins=num_bins,
                        range=x_range,
                        density=True,
                    )
                    y_max = max(y_max, max(weights))

    values = sorted(data.keys(), key=lambda x: float(x))

    def plot_func(n, illicit_exit, delta):
        one_run_data = data[n][illicit_exit][delta]
        if show_estimated_effects:
            estimated_effects = one_run_data["estimated_effects"]
            title = "True and Estimated Heterogeneous Effects"
        else:
            estimated_effects = None
            title = "True Effects"
        title += "\nSample Size n = {}".format(n)
        title += '\nIllicit User "Exit Rate" = {}'.format(illicit_exit)
        title += "\nTreatment Nonmedical Incidence Delta = {}".format(delta)
        effects_histogram(
            one_run_data["true_effects"],
            estimated_effects,
            num_bins=num_bins,
            title=title,
            x_range=x_range,
            y_max=y_max,
            show_avg=False,
        )

    sample_sizes = sorted(data.keys(), key=lambda x: int(x))
    illicit_exits = sorted(data[sample_sizes[0]].keys(), key=lambda x: float(x))
    deltas = sorted(
        data[sample_sizes[0]][illicit_exits[0]].keys(), key=lambda x: float(x)
    )

    sample_size_widget = widgets.SelectionSlider(
        options=sample_sizes,
        value=sample_sizes[0],
        description="n",
        continuous_update=False,
    )
    illicit_exit_widget = widgets.SelectionSlider(
        options=illicit_exits,
        value=illicit_exits[len(illicit_exits) // 2],
        description="illicit_exit",
        continuous_update=False,
    )
    delta_widget = widgets.SelectionSlider(
        options=deltas,
        value=deltas[len(deltas) // 2],
        description="delta",
        continuous_update=False,
    )

    ui = widgets.VBox([sample_size_widget, illicit_exit_widget, delta_widget])
    out = widgets.interactive_output(
        plot_func,
        {
            "n": sample_size_widget,
            "illicit_exit": illicit_exit_widget,
            "delta": delta_widget,
        },
    )
    display(ui, out)


def plot_relative_error(data_dir="heterogeneous_example_data"):
    if data_dir[-1] != "/":
        data_dir += "/"
    data_filename = data_dir + "heterogeneous_example_error_data.json"
    data = json.load(open(data_filename))

    plotting_data = []
    for n in data.keys():
        for i in data[n].keys():
            d = {"Sample Size": int(n), "Iteration": int(i)}
            true_effects = data[n][str(i)]["true_effects"]
            estimated_effects = data[n][str(i)]["estimated_effects"]
            error = np.array(true_effects) - np.array(estimated_effects)
            relative_error = np.linalg.norm(error) / np.linalg.norm(true_effects)
            d["Relative Error"] = relative_error
            plotting_data.append(d)
    plotting_df = pd.DataFrame(plotting_data)

    plt.figure(figsize=(18, 8))
    ax = plt.gca()
    grid = sns.lineplot(
        x="Sample Size", y="Relative Error", data=plotting_df, marker="o", ax=ax
    )
    sample_sizes = [int(n) for n in data.keys()]
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels([""] + sample_sizes[1:], rotation=45)
    ax.set_xlim([0, max(sample_sizes) + 100])
    sns.despine()
    plt.title("Relative Error of Causal Forest")
    plt.show()
