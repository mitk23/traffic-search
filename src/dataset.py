import torch


class STDataset(torch.utils.data.Dataset):
    n_up, n_down = 32, 31

    def __init__(
        self,
        X,
        y,
        time_step,
        prediction_horizon=1,
        space_window=None,
        static_features=[0, 1],
    ):
        """
        Parameters
        ----------
        X:
        y:
        time_step: int
            過去を参照する時間幅 (x 15min)
        prediction_horizon: int
            いくつ先のstepを予測するか (>= 1 x 15min)
        space_window: Tuple[int]
            前後の区間をいくつ参照するか
        static_features: List[int] | Tuple[int]
            静的な特徴量の列番号を示す
        """
        assert (
            X.dim() == 3
        ), "X should be Spatial-Temporal Matrix (Sections x Periods x Features)"
        assert (
            y.dim() == 3
        ), "y should be Spatial-Temporal Matrix (Sections x Periods x 1 (label))"

        assert time_step > 0, "time step must be >0 (x15min)"
        assert prediction_horizon > 0, "prediction horizon must be >0 (x15min)"

        self.time_step = time_step
        self.space_window = space_window
        self.prediction_horizon = prediction_horizon

        assert isinstance(
            static_features, (list, tuple)
        ), "static features must be List[int] or Tuple[int]"
        assert isinstance(
            static_features[0], int
        ), "static features must be List[int] or Tuple[int]"
        self.static_features = static_features

        if space_window is not None:
            assert isinstance(
                space_window, (list, tuple)
            ), "space window must be List[int] or Tuple[int]"
            assert isinstance(
                space_window[0], int
            ), "space window must be List[int] or Tuple[int]"
            assert (
                len(space_window) == 2
            ), "space window must be (-upstream_step, downstream_step)"
            f = self.__sliding_space(X)
            features, labels = self.__sliding_window(f, y)
        else:
            features, labels = self.__sliding_window(X, y)

        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __sliding_space(self, X):
        S, T, D = X.shape
        s_window = self.space_window
        s_len = s_window[1] - s_window[0] + 1

        features = torch.empty((S, T, D, s_len), dtype=torch.float32)

        for sec_id in range(S):
            is_up = sec_id < self.n_up
            # 各区間の前後区間を切り出す
            for i, offset in enumerate(range(s_window[0], s_window[1] + 1)):
                neighbor_id = sec_id + offset
                # paddingする
                if neighbor_id < 0:
                    neighbor_id = 0
                elif neighbor_id >= S:
                    neighbor_id = S - 1
                # 上り区間からはみ出す
                elif is_up and neighbor_id >= self.n_up:
                    neighbor_id = self.n_up - 1
                # 下り区間からはみ出す
                elif (not is_up) and (neighbor_id < self.n_up):
                    neighbor_id = self.n_up

                features[sec_id, ..., i] = X[neighbor_id]

        return features

    def __sliding_window(self, X, y):
        S, T, *_ = X.shape
        t_step = self.time_step
        p_horizon = self.prediction_horizon

        features = []
        labels = []
        for i in range(S):
            # 各区間ごとに過去time step分だけ切り出す
            for t in range(t_step, T - p_horizon + 1):
                feature = X[i, t - t_step : t + p_horizon - 1]
                # staticな情報（カレンダー, 区間）は固定する
                # feature[..., self.static_features] = X[
                #     i, t + p_horizon - 1, self.static_features
                # ]
                label = y[i, t + p_horizon - 1]
                features.append(feature)
                labels.append(label)

        features = torch.cat(features, dim=0).view(
            len(features), *features[0].shape
        )
        labels = torch.cat(labels, dim=0).view(len(labels), *labels[0].shape)

        return features, labels
