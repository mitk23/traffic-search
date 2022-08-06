import numpy as np


class STMatrixStandardScaler:
    _direction_col = 4
    _eps = 1e-8

    def __init__(self, by_up_down=False):
        self.by_up_down = by_up_down

    def fit(self, X):
        assert (
            X.dim() == 3
        ), "X should be Spatial-Temporal Matrix (Sections x Periods x Features)"
        S, T, D = X.shape

        if self.by_up_down:
            X_up = X[X[..., self._direction_col] == 0].view(-1, T, D)
            X_down = X[X[..., self._direction_col] == 1].view(-1, T, D)

            mean_up, std_up = self.__calc_params(X_up)
            mean_down, std_down = self.__calc_params(X_down)

            self.mean_ = (mean_up, mean_down)
            self.std_ = (std_up, std_down)
        else:
            mean, std = self.__calc_params(X)
            self.mean_ = mean
            self.std_ = std

    def transform(self, X):
        S, T, D = X.shape

        if self.by_up_down:
            X_up = X[X[..., self._direction_col] == 0].view(-1, T, D)
            X_down = X[X[..., self._direction_col] == 1].view(-1, T, D)

            X_up_norm = self.__transform(X_up, self.mean_[0], self.std_[0])
            X_down_norm = self.__transform(X_down, self.mean_[1], self.std_[1])

            X_norm = np.vstack((X_up_norm, X_down_norm))
        else:
            X_norm = self.__transform(X, self.mean_, self.std_)

        return X_norm

    def fit_transform(self, X):
        self.fit(X)
        X_norm = self.transform(X)
        return X_norm

    def inverse_transform(self, X):
        X_origin = X * (self.std_ + self._eps) + self.mean_
        return X_origin

    def get_params(self):
        params = {"mean": self.mean_, "std": self.std_}
        return params

    def __calc_params(self, X):
        S, T, D = X.shape
        mean = X.reshape(-1, D).mean(axis=0)
        std = X.reshape(-1, D).std(axis=0)
        return mean, std

    def __transform(self, X, mean, std):
        X_norm = (X - mean) / (std + self._eps)
        return X_norm
