import torch.nn as nn

from config import hparams

from .traffic_volume_encoder import TemporalTrafficVolumeEncoder, TrafficVolumeEncoder


class SpecifiedSearchCountEncoder(TrafficVolumeEncoder):
    def __init__(self):
        super().__init__()

        # hyper params
        self.conv_dim = hparams.SPECIFIED_SEARCH_COUNT_CONV_DIM
        self.conv_kernel = hparams.SPECIFIED_SEARCH_COUNT_CONV_KERNEL
        self.conv_padding = hparams.SPECIFIED_SEARCH_COUNT_CONV_PADDING
        self.conv_padding_mode = hparams.SPECIFIED_SEARCH_COUNT_CONV_PADDING_MODE

        self.lstm_dim = hparams.SPECIFIED_SEARCH_COUNT_LSTM_DIM
        self.lstm_layers = hparams.SPECIFIED_SEARCH_COUNT_LSTM_LAYERS
        self.lstm_dropout = hparams.SPECIFIED_SEARCH_COUNT_LSTM_DROPOUT
        self.bidirectional = hparams.BIDIRECTIONAL_LSTM

        self.conv = nn.Conv2d(
            1,
            self.conv_dim,
            self.conv_kernel,
            padding=self.conv_padding,
            padding_mode=self.conv_padding_mode,
        )
        self.lstm = nn.LSTM(
            self.conv_dim,
            self.lstm_dim,
            num_layers=self.lstm_layers,
            dropout=self.lstm_dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = super().forward(x)

        return out


class TemporalSpecifiedSearchCountEncoder(TemporalTrafficVolumeEncoder):
    def __init__(self):
        super().__init__()

        # hyper params
        self.lstm_dim = hparams.SPECIFIED_SEARCH_COUNT_LSTM_DIM
        self.lstm_layers = hparams.SPECIFIED_SEARCH_COUNT_LSTM_LAYERS
        self.lstm_dropout = hparams.SPECIFIED_SEARCH_COUNT_LSTM_DROPOUT
        self.bidirectional = hparams.BIDIRECTIONAL_LSTM

        self.lstm = nn.LSTM(
            1,
            self.lstm_dim,
            num_layers=self.lstm_layers,
            dropout=self.lstm_dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = super().forward(x)

        return out


class UnspecifiedSearchCountEncoder(TrafficVolumeEncoder):
    def __init__(self):
        super().__init__()

        # hyper params
        self.conv_dim = hparams.UNSPECIFIED_SEARCH_COUNT_CONV_DIM
        self.conv_kernel = hparams.UNSPECIFIED_SEARCH_COUNT_CONV_KERNEL
        self.conv_padding = hparams.UNSPECIFIED_SEARCH_COUNT_CONV_PADDING
        self.conv_padding_mode = hparams.UNSPECIFIED_SEARCH_COUNT_CONV_PADDING_MODE

        self.lstm_dim = hparams.UNSPECIFIED_SEARCH_COUNT_LSTM_DIM
        self.lstm_layers = hparams.UNSPECIFIED_SEARCH_COUNT_LSTM_LAYERS
        self.lstm_dropout = hparams.UNSPECIFIED_SEARCH_COUNT_LSTM_DROPOUT
        self.bidirectional = hparams.BIDIRECTIONAL_LSTM

        self.conv = nn.Conv2d(
            1,
            self.conv_dim,
            self.conv_kernel,
            padding=self.conv_padding,
            padding_mode=self.conv_padding_mode,
        )
        self.lstm = nn.LSTM(
            self.conv_dim,
            self.lstm_dim,
            num_layers=self.lstm_layers,
            dropout=self.lstm_dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = super().forward(x)

        return out


class TemporalUnspecifiedSearchCountEncoder(TemporalTrafficVolumeEncoder):
    def __init__(self):
        super().__init__()

        # hyper params
        self.lstm_dim = hparams.UNSPECIFIED_SEARCH_COUNT_LSTM_DIM
        self.lstm_layers = hparams.UNSPECIFIED_SEARCH_COUNT_LSTM_LAYERS
        self.lstm_dropout = hparams.UNSPECIFIED_SEARCH_COUNT_LSTM_DROPOUT
        self.bidirectional = hparams.BIDIRECTIONAL_LSTM

        self.lstm = nn.LSTM(
            1,
            self.lstm_dim,
            num_layers=self.lstm_layers,
            dropout=self.lstm_dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = super().forward(x)

        return out


class SearchCountEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.specified_search_count_encoder = SpecifiedSearchCountEncoder()
        self.unspecified_search_count_encoder = UnspecifiedSearchCountEncoder()

    def forward(self, x_specified, x_unspecified):
        N, T_hour, W, D_specified = x_specified.shape
        _, T_day, _, D_unspecified = x_unspecified.shape

        # N x T_hour x W x D -> (N x T_hour x H, (n_layer x N x H, n_layer x N x H))
        out_specified = self.specified_search_count_encoder(x_specified)
        # N x T_day x W x D -> (N x T_day x H, (n_layer x N x H, n_layer x N x H))
        out_unspecified = self.unspecified_search_count_encoder(x_unspecified)

        return out_specified, out_unspecified


class TemporalSearchCountEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.specified_search_count_encoder = TemporalSpecifiedSearchCountEncoder()
        self.unspecified_search_count_encoder = TemporalUnspecifiedSearchCountEncoder()

    def forward(self, x_specified, x_unspecified):
        N, T_hour, D_specified = x_specified.shape
        _, T_day, D_unspecified = x_unspecified.shape

        # N x T_hour x D -> (N x T_hour x H, (n_layer x N x H, n_layer x N x H))
        out_specified = self.specified_search_count_encoder(x_specified)
        # N x T_day x W x D -> (N x T_day x H, (n_layer x N x H, n_layer x N x H))
        out_unspecified = self.unspecified_search_count_encoder(x_unspecified)

        return out_specified, out_unspecified
