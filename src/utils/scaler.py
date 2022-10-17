import torch


class STMatrixStandardScaler:
    _direction_thresh = 32
    _search_col = [-3, -2]
    _eps = 1e-5

    def __init__(
        self,
        road_col=1,
        search_col=None,
        traffic_col=None,
        by_up_down=False,
        skip_features=None,
    ):
        self.by_up_down = by_up_down
        self.skip_features = skip_features

    def fit(self, X):
        assert (
            X.dim() == 3
        ), "X should be Spatial-Temporal Matrix (Features x Periods x Sections)"
        D, T, _ = X.shape

        if self.by_up_down:
            X_up = X[..., X[1] < self._direction_thresh].view(D, T, -1)
            X_down = X[..., X[1] >= self._direction_thresh].view(D, T, -1)

            mean_up, std_up = self.__calc_params(X_up)
            mean_down, std_down = self.__calc_params(X_down)

            self.mean_ = (mean_up, mean_down)
            self.std_ = (std_up, std_down)
        else:
            mean, std = self.__calc_params(X)
            self.mean_ = mean
            self.std_ = std

    def transform(self, X):
        D, T, _ = X.shape

        if self.by_up_down:
            X_up = X[..., X[1] < self._direction_thresh].view(D, T, -1)
            X_down = X[..., X[1] >= self._direction_thresh].view(D, T, -1)

            X_up_norm = self.__transform(X_up, self.mean_[0], self.std_[0])
            X_down_norm = self.__transform(X_down, self.mean_[1], self.std_[1])

            X_norm = torch.cat([X_up_norm, X_down_norm], dim=2)
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
        D, *_ = X.shape
        mean = X.view(D, -1).mean(dim=1)
        std = X.view(D, -1).std(dim=1)
        return mean, std

    def __transform(self, X, mean, std):
        X_norm = (X - mean[:, None, None]) / (std[:, None, None] + self._eps)

        if self.skip_features is not None:
            assert isinstance(
                self.skip_features, (list, tuple)
            ), "skip features must be List[int] | Tuple[int]"
            assert isinstance(
                self.skip_features[0], int
            ), "skip features must be List[int] | Tuple[int]"
            X_norm[self.skip_features] = X[self.skip_features]

        return X_norm
