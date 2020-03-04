from numba import jit
import numpy as np


@jit(nopython=True)
def evaluate_loss(X, Y, theta, lam):
    n = X.shape[0]

    X_perf = np.copy(X)

    # compute log likelihood
    t1 = (
        1.0
        / n
        * np.sum(
            -1.0 * np.multiply(Y, X_perf @ theta) + np.log(1 + np.exp(X_perf @ theta))
        )
    )

    # add regularization (without considering the bias)
    t2 = lam / 2.0 * np.linalg.norm(theta[:-1]) ** 2
    loss = t1 + t2

    return loss


@jit(nopython=True)
def logistic_regression(X_orig, Y_orig, lam, method, tol=1e-7, theta_init=None):

    # assumes that the last coordinate is the bias term
    X = np.copy(X_orig)
    Y = np.copy(Y_orig)
    n, d = X.shape

    # compute smoothness of the logistic loss
    smoothness = np.sum(np.square(np.linalg.norm(X, axis=1))) / (4.0 * n)

    if method == "Exact":
        eta_init = 1 / (smoothness + lam)  # true smoothness

    elif method == "GD":
        assert theta_init is not None
        eta_init = 2 / (smoothness + 2 * lam)

    else:
        print("method must be Exact or GD")
        raise ValueError

    if theta_init is not None:
        theta = np.copy(theta_init)
    else:
        theta = np.zeros(d)

    # evaluate initial loss
    prev_loss = evaluate_loss(X, Y, theta, lam)

    loss_list = [prev_loss]
    is_gd = False
    i = 0
    gap = 1e30

    eta = eta_init

    while gap > tol and not is_gd:

        # take gradients
        exp_tx = np.exp(X @ theta)
        c = exp_tx / (1 + exp_tx) - Y
        gradient = 1.0 / n * np.sum(X * c[:, np.newaxis], axis=0) + lam * np.append(
            theta[:-1], 0
        )

        new_theta = theta - eta * gradient

        # compute new loss
        loss = evaluate_loss(X, Y, new_theta, lam)

        # do backtracking line search
        if loss > prev_loss and method == "Exact":
            eta = eta * 0.1
            gap = 1e30
            continue
        else:
            eta = eta_init

        theta = np.copy(new_theta)

        loss_list.append(loss)
        gap = prev_loss - loss
        prev_loss = loss

        if method == "GD":
            is_gd = True

        i += 1

    return theta, loss_list, smoothness
