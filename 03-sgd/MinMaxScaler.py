import numpy as np
import  sklearn

class MinMaxScaler:
    def __init__(self, *, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X):
        self.n_samples_seen_ = X.shape[1]
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.scale_ = (max(self.feature_range) - min(self.feature_range)) / np.array([
            x if x != 0 else 1 for x in self.data_range_])
        self.min_ = min(self.feature_range) - self.data_min_ * self.scale_
        return self

    def transform(self, X):
        X *= self.scale_
        X += self.min_
        return X
    def fit_transform(self, X):
        return self.fit(X).transform(X)
