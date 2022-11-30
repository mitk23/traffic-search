import torch.nn as nn
import torch.nn.functional as F

from config import hparams


class TrafficVolumeEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # hyper params
        self.conv_dim = hparams.TRAFFIC_VOLUME_CONV_DIM
        self.conv_kernel = hparams.TRAFFIC_VOLUME_CONV_KERNEL
        self.conv_padding = hparams.TRAFFIC_VOLUME_CONV_PADDING
        self.conv_padding_mode = hparams.TRAFFIC_VOLUME_CONV_PADDING_MODE

        self.lstm_dim = hparams.TRAFFIC_VOLUME_LSTM_DIM
        self.lstm_layers = hparams.TRAFFIC_VOLUME_LSTM_LAYERS
        self.lstm_dropout = hparams.TRAFFIC_VOLUME_LSTM_DROPOUT
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
        N, T, W, D = x.shape

        # N x T x W x D -> N x D x T x W
        out = x.permute(0, 3, 1, 2)
        # Convolution: N x D x T x W -> N x C x T x 1
        out = F.relu(self.conv(out))
        # N x C x T x 1 -> N x C x T
        out = out[..., 0]
        # N x C x T -> N x T x C
        out = out.permute(0, 2, 1)
        # LSTM: N x T x C -> N x T x H, (n_layer x N x H, n_layer x N x H)
        out, state = self.lstm(out)

        # changing state shape from bi-directional to uni-directional
        h, c = state
        N_LAYER, _, H = h.shape
        # n_layer x N x H -> N x n_layer x H
        h = h.transpose(0, 1)
        c = c.transpose(0, 1)
        # N x n_layer x H -> N x (n_layer/2) x (2H)
        h = h.reshape(N, N_LAYER // 2, 2 * H)
        c = c.reshape(N, N_LAYER // 2, 2 * H)
        # N x (n_layer/2) x (2H)
        h = h.transpose(0, 1).contiguous()
        c = c.transpose(0, 1).contiguous()
        state = (h, c)

        return out, state


class TemporalTrafficVolumeEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # hyper params
        self.lstm_dim = hparams.TRAFFIC_VOLUME_LSTM_DIM
        self.lstm_layers = hparams.TRAFFIC_VOLUME_LSTM_LAYERS
        self.lstm_dropout = hparams.TRAFFIC_VOLUME_LSTM_DROPOUT
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
        N, T, D = x.shape

        # LSTM: N x T x D -> N x T x H, (n_layer x N x H, n_layer x N x H)
        out, state = self.lstm(x)

        # changing state shape from bi-directional to uni-directional
        h, c = state
        N_LAYER, _, H = h.shape
        # n_layer x N x H -> N x n_layer x H
        h = h.transpose(0, 1)
        c = c.transpose(0, 1)
        # N x n_layer x H -> N x (n_layer/2) x (2H)
        h = h.reshape(N, N_LAYER // 2, 2 * H)
        c = c.reshape(N, N_LAYER // 2, 2 * H)
        # N x (n_layer/2) x (2H)
        h = h.transpose(0, 1).contiguous()
        c = c.transpose(0, 1).contiguous()
        state = (h, c)

        return out, state
