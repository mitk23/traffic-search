import numpy as np
import torch

from config import config, hparams


class STDataset(torch.utils.data.Dataset):
    FEATURE_NAMES = [
        "traffic",
        "specified_search",
        "unspecified_search",
        "weather",
        "road",
        "datetime",
    ]

    def __init__(
        self,
        feature_stmatrix_dict,
        label_stmatrix,
        time_step=hparams.TIME_STEP,
        prediction_horizon=hparams.PREDICTION_HORIZON,
        prediction_interval=hparams.PREDICTION_INTERVAL,
        neighbor_window=None,
    ):
        self.time_step = time_step
        self.prediction_horizon = prediction_horizon
        self.prediction_interval = prediction_interval
        self.neighbor_window = neighbor_window

        for f_name in self.FEATURE_NAMES:
            assert (
                f_name in feature_stmatrix_dict
            ), f"feature_stmatrix_dict must contain {self.FEATURE_NAMES}"

        self.feature_stmatrix_dict = feature_stmatrix_dict
        self.label_stmatrix = label_stmatrix

        slided_feature_dict, slided_labels = self.sliding_spatio_temporal_window(
            feature_stmatrix_dict, label_stmatrix
        )
        assert all(
            feature.shape[0] == slided_feature_dict["traffic"].shape[0]
            for feature in slided_feature_dict.values()
        )

        self.feature_dict = slided_feature_dict
        self.labels = slided_labels

    def __len__(self):
        return self.feature_dict["traffic"].shape[0]

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.feature_dict.items()}, self.labels[index]

    def sliding_spatio_temporal_window(self, feature_stmatrix_dict, label_stmatrix):
        slided_feature_list_dict, slided_label_list = self.__sliding_time_window(
            feature_stmatrix_dict, label_stmatrix
        )

        slided_traffic_list = slided_feature_list_dict["traffic"]
        slided_specified_search_list = slided_feature_list_dict["specified_search"]
        slided_unspecified_search_list = slided_feature_list_dict["unspecified_search"]
        slided_weather_list = slided_feature_list_dict["weather"]
        slided_road_list = slided_feature_list_dict["road"]
        slided_datetime_list = slided_feature_list_dict["datetime"]

        # formatting the dimension of each tensor
        if self.neighbor_window is None:
            # N_time x (D x T_hour x S) -> N x T_hour x D
            traffic = torch.cat(slided_traffic_list, dim=2).transpose(2, 0).contiguous()

            # N_time x (D x P_hour x S) -> N x P_hour x D
            specified_search = (
                torch.cat(slided_specified_search_list, dim=2)
                .transpose(2, 0)
                .contiguous()
            )

            # N_time x (D x P_day x S) -> N x P_day x D
            unspecified_search = (
                torch.cat(slided_unspecified_search_list, dim=2)
                .transpose(2, 0)
                .contiguous()
            )
            weather = torch.cat(slided_weather_list, dim=2).transpose(2, 0).contiguous()
        else:
            # N_time x (D x T_hour x S) -> N x D x T_hour x W -> N x T_hour x W x D
            traffic = self.__append_neighbor_feature(slided_traffic_list)
            traffic = traffic.permute(0, 2, 3, 1).contiguous()

            # N_time x (D x P_hour x S) -> N x D x P_hour x W -> N x P_hour x W x D
            specified_search = self.__append_neighbor_feature(
                slided_specified_search_list
            )
            specified_search = specified_search.permute(0, 2, 3, 1).contiguous()

            # N_time x (D x P_day x S) -> N x D x P_day x W -> N x P_day x W x D
            unspecified_search = self.__append_neighbor_feature(
                slided_unspecified_search_list
            )
            unspecified_search = unspecified_search.permute(0, 2, 3, 1).contiguous()

            weather = self.__append_neighbor_feature(slided_weather_list)
            weather = weather.permute(0, 2, 3, 1).contiguous()

        # N_time x (1 x 1 x S) -> N x 1 x 1
        road = torch.cat(slided_road_list, dim=2).transpose(2, 0).contiguous()
        # N_time x (1 x P_hour x 1) -> N_time x (1 x P_hour x S) -> N x P_hour x 1
        S = slided_label_list[0].shape[-1]
        slided_datetime_list = [
            slided_datetime.repeat(1, 1, S) for slided_datetime in slided_datetime_list
        ]
        datetime = torch.cat(slided_datetime_list, dim=2).transpose(2, 0).contiguous()

        # N_time x (1 x P_hour x S) -> N x P_hour x 1
        slided_labels = torch.cat(slided_label_list, dim=2).transpose(2, 0).contiguous()

        slided_feature_dict = {
            "traffic": traffic,
            "specified_search": specified_search,
            "unspecified_search": unspecified_search,
            "weather": weather,
            "road": road,
            "datetime": datetime,
        }
        return slided_feature_dict, slided_labels

    def __sliding_time_window(self, feature_stmatrix_dict, label_stmatrix):
        """
        cut out tensors to the time step size for traffic prediction

        Parameters
        ----------
        feature_stmatrix_dict: Dict[str, torch.Tensor]
            each feature name mapped to its spatio-temporal matrix
            size: n_feature D x n_time_length L x n_segment S
        label_stmatrix: torch.Tensor
            spatio-temporal matrix of labels
        """
        # D x L_hour x S
        t_traffic = feature_stmatrix_dict["traffic"]
        t_specified_search = feature_stmatrix_dict["specified_search"]
        # D x L_day x S
        t_unspecified_search = feature_stmatrix_dict["unspecified_search"]
        t_weather = feature_stmatrix_dict["weather"]
        # 1 x 1 x S (time-static)
        t_road = feature_stmatrix_dict["road"]
        # 1 x L_hour x 1
        t_datetime = feature_stmatrix_dict["datetime"]

        # D x L_hour x S
        t_label = label_stmatrix

        # List[torch.Tensor(D x T_hour / P_hour / P_day x S)]
        slided_traffic_list = []
        slided_specified_search_list = []
        slided_unspecified_search_list = []
        slided_weather_list = []
        slided_datetime_list = []
        slided_label_list = []

        _, L_hour, S = t_traffic.shape

        for t in range(
            self.time_step,
            L_hour - self.prediction_horizon + 1,
            self.prediction_interval,
        ):
            # hourly features / labels
            slided_traffic = t_traffic[:, t - self.time_step : t, :]
            slided_traffic_list.append(slided_traffic)

            slided_specified_search = t_specified_search[
                :, t : t + self.prediction_horizon, :
            ]
            slided_specified_search_list.append(slided_specified_search)

            slided_datetime = t_datetime[:, t : t + self.prediction_horizon, :]
            slided_datetime_list.append(slided_datetime)

            label = t_label[:, t : t + self.prediction_horizon, :]
            slided_label_list.append(label)

            # daily features
            t_day = t // 24
            slided_unspecified_search = t_unspecified_search[
                :, t_day : t_day + (self.prediction_horizon // 24), :
            ]
            slided_unspecified_search_list.append(slided_unspecified_search)

            slided_weather = t_weather[
                :, t_day : t_day + (self.prediction_horizon // 24), :
            ]
            slided_weather_list.append(slided_weather)

        # time-static features
        N_time_length = len(slided_label_list)
        slided_road = [t_road for _ in range(N_time_length)]

        slided_feature_list_dict = {
            "traffic": slided_traffic_list,
            "specified_search": slided_specified_search_list,
            "unspecified_search": slided_unspecified_search_list,
            "weather": slided_weather_list,
            "road": slided_road,
            "datetime": slided_datetime_list,
        }
        return slided_feature_list_dict, slided_label_list

    def __append_neighbor_feature(self, slided_feature_list):
        """
        get the feature of neighbor segments together at certain time step and segment

        Parameters
        ----------
        slided_feature_list: List[torch.Tensor]
            list of feature tensors (size: n_feature D x n_timestep T x n_segment S)
            each tensor contains feature data about all segments on every time steps
        neighbor_window: Tuple[int]
            pair of offsets utilized to calculate the index of neighbor segments
            e.g.) for i-th segment, neighbor segments are defined to be from (i - neighbor_window[0])-th to (i + neighbor_window[1])-th segment.

        Returns
        -------
        st_slided_feature: torch.Tensor
            tensor size: n_sample N x n_feature D x n_timestep T x n_neighbor_segment W
        """
        D, T, S = slided_feature_list[0].shape
        N_SAMPLE = len(slided_feature_list) * S

        neighbor_iterator = list(
            range(self.neighbor_window[0], self.neighbor_window[1] + 1)
        )
        N_NEIGHBOR = len(neighbor_iterator)

        st_slided_feature = torch.empty(N_SAMPLE, D, T, N_NEIGHBOR)

        i_sample = 0
        for feature in slided_feature_list:
            for sec_id in range(S):
                for i_neighbor, offset in enumerate(neighbor_iterator):
                    neighbor_sec_id = sec_id + offset

                    if sec_id < config.N_INBOUND:  # 上り区間からはみ出す場合
                        neighbor_sec_id = np.clip(
                            neighbor_sec_id, 0, config.N_INBOUND - 1
                        )
                    else:  # 下り区間からはみ出す場合
                        neighbor_sec_id = np.clip(
                            neighbor_sec_id, config.N_INBOUND, S - 1
                        )

                    st_slided_feature[i_sample, ..., i_neighbor] = feature[
                        ..., neighbor_sec_id
                    ]

                i_sample += 1

        return st_slided_feature
