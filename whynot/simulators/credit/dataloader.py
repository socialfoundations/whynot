"""Load and preprocess Kaggle credit dataset."""
import os

import numpy as np
import pandas as pd
from sklearn import preprocessing


class DataLoader(object):
    """Class to lazily load the credit dataset."""

    def __init__(self, datafile="credit_data.zip", seed=None):
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        datapath = os.path.join(cur_dir, datafile)

        self.datapath = datapath
        self._features = None
        self._labels = None
        self.seed = seed

    @property
    def features(self):
        """Return the dataset features."""
        if self._features is None:
            self._features, self._labels = self.load_data()
        return np.copy(self._features)

    @property
    def labels(self):
        """Return the dataset labels."""
        if self._labels is None:
            self._features, self._labels = self.load_data()
        return np.copy(self._labels)

    @property
    def num_agents(self):
        """Compute number of agents in the dataset."""
        return self.features.shape[0]

    @property
    def num_features(self):
        """Compute number of features for each agent."""
        return self.features.shape[1]

    def load_data(self):
        """Load, preprocess and class-balance the credit data."""
        rng = np.random.RandomState(self.seed)

        data = pd.read_csv(self.datapath, index_col=0)
        data.dropna(inplace=True)

        features = data.drop("SeriousDlqin2yrs", axis=1)
        # zero mean, unit variance
        features = preprocessing.scale(features)

        # add bias term
        features = np.append(features, np.ones((features.shape[0], 1)), axis=1)
        outcomes = np.array(data["SeriousDlqin2yrs"])

        # balance classes
        default_indices = np.where(outcomes == 1)[0]
        other_indices = np.where(outcomes == 0)[0][:10000]
        indices = np.concatenate((default_indices, other_indices))

        features_balanced = features[indices]
        outcomes_balanced = outcomes[indices]

        shape = features_balanced.shape

        # shuffle arrays
        shuffled = rng.permutation(len(indices))
        return features_balanced[shuffled], outcomes_balanced[shuffled]


CreditData = DataLoader(seed=0)
