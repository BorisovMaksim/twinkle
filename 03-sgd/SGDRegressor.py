import typing as ty

import numpy as np

import metrics
from _losses import LossFunction, SquaredLoss
from _sgd import sgd


class SGDRegressor():
    def __init__(
            self,
            loss: str = 'squared_loss',
            fit_intercept: bool = True,
            max_iter: int = 1000,
            eta0: float = 0.01,
            shuffle: bool = True,
            verbose: bool = False,
            seed: int = 1,
            random_state: int = None,
            warm_start: bool = False,
            coef_init: np.array = None,
            intercept_init: np.array = None
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
        self.random_state = random_state
        self.warm_start = warm_start
        self.coef_init = coef_init
        self.intercept_init = intercept_init

        # Validate saved params in method below
        self.__validate_params()

    def __validate_params(self):
        if not isinstance(self.fit_intercept, bool):
            raise ValueError(f'fit_intercept {self.fit_intercept} has to be bool')
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError(f'fit_intercept {self.fit_intercept} has to be bool')

    def fit(self, X, y, sample_weight=None):
        return self._partial_fit(X, y, epoch=self.max_iter, sample_weight=sample_weight)

    def partial_fit(self, X, y, sample_weight=None):
        return self._partial_fit(X, y, epoch=1, sample_weight=sample_weight)

    def _partial_fit(self, X, y, epoch, sample_weight=None):
        if self.warm_start:
            if self.coef_init is not None and self.intercept_init is not None:
                weights = self.coef_init
                intercept= self.intercept_init
        else:
            weights = np.ones(len(X[0]))
            intercept = np.array(int(self.fit_intercept))
            num_epoch = 0

            try:
                weights = self.coef_
                intercept = self.intercept_
                num_epoch = self.epoch_
                print("WEIGHTS UPDATED")
            except AttributeError:
                print("WEIGHTS NOT UPDATED")


        weights, intercept, epoch = sgd(X=X, y=y, fit_intercept=self.fit_intercept, max_iter=epoch,
                                        eta0=self.eta0,
                                        weights=weights,
                                        intercept=intercept, loss=self.__loss,
                                        sample_weight=sample_weight, seed=self.random_state,
                                        shuffle=self.shuffle, verbose=self.verbose)
        self.coef_ = weights
        self.intercept_ = np.array([intercept])
        self.epoch_ = epoch + num_epoch

        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        return metrics.r2_score(y_true=y, y_pred=y_pred)
