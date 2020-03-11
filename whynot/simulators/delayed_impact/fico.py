"""Utilities for loading and manipulating FICO data.

Borrowed from:
    https://github.com/ecreager/delayedimpact-scm-shareable
"""
import os
from typing import Callable, Iterable

import numpy as np
import pandas as pd

Index = pd.Index
Array = np.ndarray
DataFrame = pd.DataFrame


#############################
# FICO Data loading
#############################

PERF = "transrisk_performance_by_race_ssa.csv"
CDF_BY_RACE = "transrisk_cdf_by_race_ssa.csv"
OVERALL = "totals.csv"

FILES = dict(cdf_by_race=CDF_BY_RACE, performance_by_race=PERF, overview=OVERALL)


def cleanup_frame(frame):
    """Make the columns have better names, and ordered in a better order."""
    frame = frame.rename(columns={"Non- Hispanic white": "White"})
    frame = frame.reindex(columns=["Asian", "White", "Hispanic", "Black"])
    return frame


def read_totals(data_dir):
    """Read the total number of people of each race."""
    frame = cleanup_frame(
        pd.read_csv(os.path.join(data_dir, FILES["overview"]), index_col=0)
    )
    return {r: frame[r]["SSA"] for r in frame.columns}


def convert_percentiles(idx):
    """Convert percentiles."""
    pdf = [
        (300, 2.1),
        (350, 4.2),
        (400, 5.4),
        (450, 6.5),
        (500, 7.9),
        (550, 9.6),
        (600, 12.0),
        (650, 13.8),
        (700, 17.0),
        (750, 15.8),
        (800, 5.7),
        (850, 0),
    ]

    def convert_one(x):
        partial = 0
        for ((v, s), (v2, _)) in zip(pdf, pdf[1:]):
            if partial + s >= x:
                return v + (v2 - v) * (x - partial) / s
            partial += s

    return np.array(list(map(convert_one, idx)))


def parse_data(data_dir, filenames=None):
    """Parse sqf data set."""
    if filenames is None:
        filenames = [FILES["cdf_by_race"], FILES["performance_by_race"]]

    cdfs = cleanup_frame(pd.read_csv(os.path.join(data_dir, filenames[0]), index_col=0))
    performance = 100 - cleanup_frame(
        pd.read_csv(os.path.join(data_dir, filenames[1]), index_col=0)
    )
    return (cdfs / 100.0, performance / 100.0)


def get_FICO_data(data_dir, do_convert_percentiles=True):
    """Get FICO data in desired format."""
    data_pair = parse_data(data_dir)
    totals = read_totals(data_dir)

    if do_convert_percentiles:
        for v in data_pair:
            v.index = convert_percentiles(v.index)
    cdfs = data_pair[0]
    performance = data_pair[1]
    return cdfs, performance, totals


########################
# Manipulating FICO data
########################


def _find_nearest_indices(idx: Index, targets: Array) -> Array:
    """Convert 1d array into array of nearest index values."""
    idx = np.asarray(idx)
    idx_ = np.expand_dims(idx, -1)
    targets = np.expand_dims(targets, 0)
    i = (np.abs(idx_ - targets)).argmin(axis=0)
    return idx[i]


def _get_pmf(cdf):
    """Convert CDF into PMF."""
    pis = np.zeros(cdf.size)
    pis[0] = cdf[0]
    for score in range(cdf.size - 1):
        pis[score + 1] = cdf[score + 1] - cdf[score]
    return pis


def _loan_repaid_probs_factory(
    repay_df: DataFrame, scores: Index
) -> Iterable[Callable[[Array], Array]]:
    """Given performance pd.DataFrame, construct mapping from X to p(Y|X)."""

    def repaid_probs_fn(query_scores: Array) -> Array:
        if isinstance(query_scores, np.ndarray):
            nearest_scores = _find_nearest_indices(scores, query_scores)
            return repay_df[nearest_scores].values

        query_score = query_scores
        nearest_score = scores[scores.get_loc(query_score, method="nearest")]
        return repay_df[nearest_score]

    return repaid_probs_fn


def get_data_args(data_dir="data"):
    """Return objects that specify p(A), p(X|A), p(Y|X,A)."""

    all_cdfs, performance, totals = get_FICO_data(data_dir)
    # NOTE: we drop last column to make the CDFs invertible ####################
    all_cdfs = all_cdfs.drop(all_cdfs.index[-1])
    performance = performance.drop(performance.index[-1])
    ############################################################################
    cdfs = all_cdfs[["White", "Black"]]

    cdf_B = cdfs["White"].values
    cdf_A = cdfs["Black"].values

    repay_B = performance["White"]
    repay_A = performance["Black"]

    scores = cdfs.index
    scores_list = scores.tolist()

    # to populate group distributions
    # to get (batchable) loan repay probabilities for a given score
    loan_repaid_probs = [
        _loan_repaid_probs_factory(repay_A, scores),
        _loan_repaid_probs_factory(repay_B, scores),
    ]

    # to get score at a given probabilities
    inv_cdfs = get_inv_cdf_fns(cdfs)

    # get probability mass functions of each group
    pi_A = _get_pmf(cdf_A)
    pi_B = _get_pmf(cdf_B)
    pis = np.vstack([pi_A, pi_B])

    # demographic statistics
    group_ratio = np.array((totals["Black"], totals["White"]))
    group_size_ratio = group_ratio / group_ratio.sum()

    rate_indices = (list(reversed(1 - cdf_A)), list(reversed(1 - cdf_B)))

    return inv_cdfs, loan_repaid_probs, pis, group_size_ratio, scores_list, rate_indices


def get_inv_cdf_fns(cdfs: DataFrame) -> Iterable[Callable[[Array], Array]]:
    """Convert DataFrame of cdfs into list of (batched) inv. cdf lambda fns."""

    def inv_cdf_factory(cdfs_df: DataFrame, key: str) -> Callable[[Array], Array]:
        """Given cdfs pd.DataFrame & key=A, make mapping from P(Y|X,A) to X."""
        series = pd.Series(cdfs_df[key].index.values, index=cdfs[key].values)
        index = series.index

        def repaid_probs_fn(query_probs: Array) -> Array:
            if isinstance(query_probs, np.ndarray):
                nearest_scores = _find_nearest_indices(index, query_probs)
                return series[nearest_scores].values

            query_prob = query_probs
            nearest_prob = index[index.get_loc(query_prob, method="nearest")]
            return series[nearest_prob]

        return repaid_probs_fn

    inv_cdfs = [inv_cdf_factory(cdfs, "Black"), inv_cdf_factory(cdfs, "White")]

    return inv_cdfs


def get_marginal_loan_repaid_probs(data_dir="data"):
    """Return object that specifies distn p(Y|X)."""

    all_cdfs, performance, totals = get_FICO_data(data_dir)
    # NOTE: we drop last column to make the CDFs invertible ####################
    all_cdfs = all_cdfs.drop(all_cdfs.index[-1])
    performance = performance.drop(performance.index[-1])
    ############################################################################
    cdfs = all_cdfs[["White", "Black"]]

    repay_B = performance["White"]
    repay_A = performance["Black"]

    scores = cdfs.index

    # demographic statistics
    group_ratio = np.array((totals["Black"], totals["White"]))
    group_size_ratio = group_ratio / group_ratio.sum()
    p_A, p_B = group_size_ratio

    repay_marginal = repay_A * p_A + repay_B * p_B
    loan_repaid_probs = _loan_repaid_probs_factory(repay_marginal, scores)
    loan_repaid_probs = [
        loan_repaid_probs,
        loan_repaid_probs,
    ]  # return a copy for each group for compatability with other utils

    return loan_repaid_probs
