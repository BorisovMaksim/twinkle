import numpy as np
from sklearn.preprocessing import StandardScaler as scaler


class StandardScaler:
    def __init__(self, *, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.var_ = None

    def fit(self, X):
        self.n_samples_seen_ = X.shape[1]
        if self.with_mean and self.with_std:
            self.mean_ = X.mean(axis=0)
            self.var_ = X.var(axis=0)
            self.scale_ = self.var_ ** (0.5)
            self.scale_[self.scale_ == 0] = 1
        elif self.with_mean:
            self.mean_ = X.mean(axis=0)
            self.scale_ = None
            self.var_ = None
        elif self.with_std:
            self.mean_ = None
            self.var_ = X.var(axis=0)
            self.scale_ = self.var_ ** (0.5)
            self.scale_[self.scale_ == 0] = 1
        else:
            self.mean_ = None
            self.scale_ = None
            self.var_ = None
        return self

    def transform(self, X):
        if self.with_mean and self.with_std:
            return (X - self.mean_) / self.scale_
        elif self.with_mean:
            return  X - self.mean_
        elif self.with_std:
            return X /self.scale_
        else:
            return X


    def fit_transform(self, X):
        return self.fit(X).transform(X)
