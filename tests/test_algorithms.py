"""Tests for causal inference methods."""
import numpy as np

import whynot as wn


def test_suite():
    """Test whether or not the entire suite runs as expected."""
    num_samples = 200
    num_features = 15

    covariates = np.random.randn(num_samples, num_features)
    treatment = (np.random.rand(num_samples) < 0.5).astype(np.int64)
    outcomes = np.random.randn(num_samples)

    _ = wn.causal_suite(covariates, treatment, outcomes)
