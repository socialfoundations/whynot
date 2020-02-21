import numpy as np


# implements best response function for agents, linear utilities quadratic costs
def best_response(X, theta, epsilon, strat_features):

    n = X.shape[0]

    X_strat = np.copy(X)

    for i in range(n):
        # move everything by epsilon in the direction towards better classification
        theta_strat = theta[strat_features]
        X_strat[i, strat_features] += -epsilon * theta_strat

    return X_strat
