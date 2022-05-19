import numpy as np


class MaxAbsScaler:
    def fit(self, X):
        self.n_samples_seen_ = len(X[0])
        self.max_abs_= abs(X).max(axis=0)
        self.scale_ = 1 / self.max_abs_

        return self

    def transform(self, X):
        X *= self.scale_
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)