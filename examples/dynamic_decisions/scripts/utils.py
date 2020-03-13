"""Utility functions for performative prediction demo."""
import numpy as np


def evaluate_logistic_loss(X, Y, theta, l2_penalty):
    """Compute the l2-penalized logistic loss function

    Parameters
    ----------
        X: np.ndarray
            A [num_samples, num_features] matrix of features. The last
            feature dimension is assumed to be the bias term.
        Y: np.ndarray
            A [num_samples] vector of binary labels.
        theta: np.ndarray
            A [num_features] vector of classifier parameters
        l2_penalty: float
            Regularization coefficient. Use l2_penalty=0 for no regularization.

    Returns
    -------
        loss: float

    """
    n = X.shape[0]

    logits = X @ theta
    log_likelihood = (
        1.0 / n * np.sum(-1.0 * np.multiply(Y, logits) + np.log(1 + np.exp(logits)))
    )

    regularization = (l2_penalty / 2.0) * np.linalg.norm(theta[:-1]) ** 2

    return log_likelihood + regularization


def fit_logistic_regression(X, Y, l2_penalty, tol=1e-7, theta_init=None):
    """Fit a logistic regression model via gradient descent.

    Parameters
    ----------
        X: np.ndarray
            A [num_samples, num_features] matrix of features.
            The last feature dimension is assumed to be the bias term.
        Y: np.ndarray
            A [num_samples] vector of binary labels.
        l2_penalty: float
            Regularization coefficient. Use l2_penalty=0 for no regularization.
        tol: float
            Stopping criteria for gradient descent
        theta_init: np.ndarray
            A [num_features] vector of classifier parameters to use a
            initialization

    Returns
    -------
        theta: np.ndarray
            The optimal [num_features] vector of classifier parameters.

    """
    X = np.copy(X)
    Y = np.copy(Y)
    n, d = X.shape

    # Smoothness of the logistic loss
    smoothness = np.sum(X ** 2) / (4.0 * n)

    # Optimal initial learning rate
    eta_init = 1 / (smoothness + l2_penalty)

    if theta_init is not None:
        theta = np.copy(theta_init)
    else:
        theta = np.zeros(d)

    # Evaluate loss at initialization
    prev_loss = evaluate_logistic_loss(X, Y, theta, l2_penalty)

    loss_list = [prev_loss]
    i = 0
    gap = 1e30

    eta = eta_init
    while gap > tol:

        # take gradients
        exp_tx = np.exp(X @ theta)
        c = exp_tx / (1 + exp_tx) - Y
        gradient = 1.0 / n * np.sum(
            X * c[:, np.newaxis], axis=0
        ) + l2_penalty * np.append(theta[:-1], 0)

        new_theta = theta - eta * gradient

        # compute new loss
        loss = evaluate_logistic_loss(X, Y, new_theta, l2_penalty)

        # do backtracking line search
        if loss > prev_loss:
            eta = eta * 0.1
            gap = 1e30
            continue

        eta = eta_init
        theta = np.copy(new_theta)

        loss_list.append(loss)
        gap = prev_loss - loss
        prev_loss = loss

        i += 1

    return theta
