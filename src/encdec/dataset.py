import torch


class STDataset(torch.utils.data.Dataset):
    n_up = 32

    def __init__(
        self,
        X,
        y,
        time_step,
        prediction_horizon=24,
        space_window=None,
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
        """
        assert (
            X.dim() == 3
        ), "X should be Spatial-Temporal Matrix (Features x Periods x Sections)"
        assert (
            y.dim() == 3
        ), "y should be Spatial-Temporal Matrix (1 (label) x Periods x Sections)"

        assert time_step > 0, "time step must be >0 (x15min)"
        assert prediction_horizon > 0, "prediction horizon must be >0 (x15min)"

        self.time_step = time_step
        self.space_window = space_window
        self.prediction_horizon = prediction_horizon

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
        return self.features[-1].shape[0]

    def __getitem__(self, index):
        return tuple(f[index] for f in self.features), self.labels[index]

    def __sliding_space(self, X):
        D, T, S = X.shape
        s_window = self.space_window
        s_len = s_window[1] - s_window[0] + 1

        features = torch.empty((D, T, S, s_len), dtype=torch.float32)

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

                features[..., sec_id, i] = X[..., neighbor_id]

        return features

    def __sliding_window(self, X, y):
        _, T, S, *_ = X.shape
        t_step = self.time_step
        p_horizon = self.prediction_horizon

        trf_t = []
        sr_t = []
        un_sr_t = []
        dt_t = []
        road_t = []
        labels = []

        for sec_id in range(S):
            # 各区間ごとに過去time step分だけ切り出す
            for t in range(t_step, T - p_horizon, p_horizon):
                f_traffic = X[-1, t - t_step : t, sec_id]
                f_search = X[-3, t : t + p_horizon, sec_id]
                f_un_search = X[-2, t : t + 1, sec_id]
                # static categorical variables
                f_dt = X[0, t : t + p_horizon, sec_id]
                if f_dt.dim() > 1:
                    f_dt = f_dt[..., 0]
                f_rd = torch.tensor([sec_id])

                label = y[:, t : t + p_horizon, sec_id]

                trf_t.append(f_traffic)
                sr_t.append(f_search)
                un_sr_t.append(f_un_search)
                dt_t.append(f_dt)
                road_t.append(f_rd)
                labels.append(label)

        trf_t = torch.cat(trf_t, dim=0).view(len(trf_t), *trf_t[0].shape)
        sr_t = torch.cat(sr_t, dim=0).view(len(sr_t), *sr_t[0].shape)
        un_sr_t = torch.cat(un_sr_t, dim=0).view(
            len(un_sr_t), *un_sr_t[0].shape
        )
        dt_t = torch.cat(dt_t, dim=0).view(len(dt_t), *dt_t[0].shape)
        road_t = torch.cat(road_t, dim=0).view(len(road_t), *road_t[0].shape)
        labels = torch.cat(labels, dim=0).view(len(labels), *labels[0].shape)

        return (dt_t, road_t, sr_t, un_sr_t, trf_t), labels
