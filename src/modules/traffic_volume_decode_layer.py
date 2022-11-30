import torch.nn as nn

from config import hparams


class TrafficVolumeDecodeLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm_dim = hparams.TRAFFIC_VOLUME_DECODE_LSTM_DIM
        self.lstm_layers = hparams.TRAFFIC_VOLUME_DECODE_LSTM_LAYERS
        self.lstm_dropout = hparams.TRAFFIC_VOLUME_DECODE_LSTM_DROPOUT

        self.bnorm = nn.BatchNorm1d(1)
        self.lstm = nn.LSTM(
            1,
            self.lstm_dim,
            num_layers=self.lstm_layers,
            dropout=self.lstm_dropout,
            batch_first=True,
        )

    def forward(self, x, state):
        # N x P x D
        N, P, D = x.shape
        # n_layer x N x H
        N_LAYER, _, H = state[0].shape

        # N x P x D -> N x D x P
        # x = x.permute(0, 2, 1)
        # Batch Normalization: N x D x P -> N x D x P
        # x = self.bnorm(x)
        # N x D x P -> N x P x D
        # x = x.permute(0, 2, 1)

        # LSTM: N x P x D -> N x P x H, (n_layer x N x H, n_layer x N x H)
        out, state = self.lstm(x, state)

        return out, state
