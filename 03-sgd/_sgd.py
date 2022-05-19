import typing as ty

import numpy as np

from _losses import LossFunction


def sgd(
    weights: np.ndarray,
    intercept: np.ndarray,
    loss: ty.Type[LossFunction],
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    fit_intercept: bool,
    verbose: bool,
    shuffle: bool,
    seed: ty.Optional[int],
    eta0: float,
    sample_weight: ty.Optional[np.ndarray],
):
    epoch = 0
    eta = eta0
    n_samples = len(X)

    for epoch in range(max_iter):
        if verbose:
            print(f'-- Epoch {epoch + 1}')

        indices =list(range(n_samples))
        if seed is not None:
            np.random.seed(seed)
        if shuffle:
            indices = np.random.permutation(indices)
        for i in indices:
            # Calculate prediction for current sample.
            y_hat = np.dot(X[i], weights) + intercept
            # Calculate squared error gradient by prediction. Use loss.dloss
            dloss = loss.dloss(p=y_hat, y=y[i])
            print_dloss(dloss, verbose)

            # Calculate prediction gradient by weights.
            dp_dw = (y_hat - y[i])*X[i]*sample_weight[i] if sample_weight is not None else (y_hat - y[i])*X[i]
            # Update weights, using gradients. Don't forget about learning rate.
            weights = weights - eta*dp_dw

            if fit_intercept:
                # Calculate prediction gradient by intercept.
                dp_dw = y_hat - y[i]
                # Update intercept, using gradients. Don't forget about learning rate.
                intercept = intercept - eta*dp_dw
    return weights, intercept, epoch + 1


def print_dloss(dloss, verbose=True):
    if verbose and dloss is not np.nan:
        if dloss >= 0:
            print(f"-- grad +{dloss:.2e}")
        else:
            print(f"-- grad {dloss:.2e}")
    pass
