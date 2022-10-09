class STMatrixStandardScaler:
    _eps = 1e-5

    def __init__(self, skip_features=None):
        self.mean_ = None
        self.std_ = None
        self.skip_features = skip_features

    def fit(self, X):
        assert (
            X.dim() == 3
        ), "X should be Spatial-Temporal Matrix (Features x Periods x Sections)"
        D, T, _ = X.shape

        mean, std = self.__calc_params(X)
        self.mean_ = mean
        self.std_ = std

    def transform(self, X):
        D, T, _ = X.shape

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
