import typing as ty

import numpy as np

from _losses import LossFunction, SquaredLoss
from _sgd import sgd


class SGDRegressor():
    def __init__(
            self,
            loss: str = 'squared_loss',
            fit_intercept: bool = True,
            max_iter: int = 1000,
            eta0: float = 0.01,
            shuffle: bool = False,
            verbose: bool = False,
            seed: int = 1
    ):
        if loss == 'squared_loss':
            self.__loss: ty.Type[LossFunction] = SquaredLoss()
        else:
            raise ValueError(f'The loss {loss} is not supported.')

        # Save constructor params
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.eta0 = eta0
        self.shuffle = shuffle
        self.verbose = verbose
        self.seed = seed

        # Validate saved params in method below
        self.__validate_params()

    def __validate_params(self):
        if not isinstance(self.fit_intercept,bool):
            raise ValueError(f'fit_intercept {self.fit_intercept} has to be bool')
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError(f'fit_intercept {self.fit_intercept} has to be bool')


    def fit(self, X, y, sample_weight=None):
        weights, intercept, epoch = sgd(X=X, y=y, fit_intercept=self.fit_intercept, max_iter=self.max_iter,
                                        eta0=self.eta0,
                                        weights=np.ones(len(X[0])),
                                        intercept=np.array(int(self.fit_intercept)), loss=self.__loss, sample_weight=sample_weight, seed=self.seed,
                                        shuffle=self.shuffle, verbose=self.verbose)
        self.coef_ = weights
        self.intercept_ = np.array([intercept])
        self.epoch_ = 1
        return self

    def partial_fit(self, X, y, sample_weight=None):
        weights, intercept, epoch = sgd(X=X, y=y, fit_intercept=self.fit_intercept, max_iter=self.max_iter,
                                        eta0=self.eta0,
                                        weights=self.coef_,
                                        intercept=self.coef_, loss=SquaredLoss, sample_weight=sample_weight, seed=1,
                                        shuffle=False, verbose=False)
        self.coef_ = weights
        self.intercept_ = np.array([intercept])
        self.epoch_ = epoch
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        # <YOUR CODE HERE>
        raise NotImplementedError()
