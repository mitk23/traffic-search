import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class CategoricalEmbedding(nn.Module):
    def __init__(self, datetime_emb_size, road_emb_size, hid_dim):
        super().__init__()

        self.datetime_emb_size = datetime_emb_size
        self.road_emb_size = road_emb_size
        self.n_emb = datetime_emb_size + road_emb_size

        self.datetime_emb = nn.Embedding(
            config.DT_TABLE_SIZE, datetime_emb_size
        )
        self.road_emb = nn.Embedding(config.SEC_TABLE_SIZE, road_emb_size)

        self.fc = nn.Linear(self.n_emb, hid_dim)

    def forward(self, x):
        x_dt = self.datetime_emb(x[..., 0].to(dtype=torch.int64))
        x_rd = self.road_emb(x[..., 1].to(dtype=torch.int64))
        x = torch.cat([x_dt, x_rd], dim=1)
        out = F.relu(self.fc(x))

        return out


class TrafficConv1dLSTM(nn.Module):
    def __init__(self, lstm_dim, kernel_size=5, num_layers=1):
        super().__init__()

        self.lstm_dim = lstm_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.conv = nn.Conv1d(config.TIME_STEP, config.TIME_STEP, kernel_size)
        self.lstm = nn.LSTM(1, lstm_dim, num_layers, batch_first=True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        _, (h, _) = self.lstm(out)
        return h[0]


class TrafficConv2dLSTM(nn.Module):
    def __init__(
        self,
        conv_dim,
        lstm_dim,
        kernel_size=(9, 3),
        padding=(4, 0),
        num_layers=1,
    ):
        super().__init__()

        self.conv_dim = conv_dim
        self.lstm_dim = lstm_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_layers = num_layers

        self.conv1 = nn.Conv2d(1, conv_dim, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(
            conv_dim, conv_dim, kernel_size, padding=padding
        )
        self.lstm = nn.LSTM(conv_dim, lstm_dim, num_layers, batch_first=True)

    def forward(self, x):
        assert x.dim() == 4, "Conv2dLSTM input size should be N x D x T x S"
        N, D, T, S = x.shape

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        # transform tensor to N x T x D
        out = out[..., 0].permute(0, 2, 1)
        assert out.shape == (
            N,
            T,
            self.conv_dim,
        ), "output size after Conv2d should be N x T x CONV_DIM"
        _, (h, _) = self.lstm(out)

        return h[0]


class SearchConv1d(nn.Module):
    def __init__(self, hid_dim, kernel_size=5):
        super().__init__()

        self.hid_dim = hid_dim
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(config.TIME_STEP, config.TIME_STEP, kernel_size)
        self.conv2 = nn.Conv1d(config.TIME_STEP, config.TIME_STEP, kernel_size)
        self.fc = nn.Linear(config.TIME_STEP * 2, hid_dim)

    def forward(self, x):
        N, D, T, S = x.shape

        out1 = F.relu(self.conv1(x[:, 0]))
        out1 = out1.view(N, -1)
        out2 = F.relu(self.conv2(x[:, 1]))
        out2 = out2.view(N, -1)

        out = torch.cat([out1, out2], dim=1)
        out = F.relu(self.fc(out))
        return out


class SearchConv2d(nn.Module):
    def __init__(self, conv_dim, hid_dim, kernel_size=(9, 3), padding=(4, 0)):
        super().__init__()

        self.conv_dim = conv_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv1 = nn.Conv2d(2, conv_dim, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(
            conv_dim, conv_dim, kernel_size, padding=padding
        )
        self.fc = nn.Linear(conv_dim * config.TIME_STEP, hid_dim)

    def forward(self, x):
        assert x.dim() == 4, "Conv2d input size should be N x D x T x S"
        N, D, T, S = x.shape
        assert D == 2, "search features length should be two"

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = out.view(N, -1)
        assert out.shape == (N, self.conv_dim * T)
        out = F.relu(self.fc(out))

        return out


class Conv1dLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.traffic = TrafficConv1dLSTM()
        self.search = SearchConv1d()

        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()

    def forward(self, x):
        assert x.dim() == 4, "historical input size should be N x D x T x S"

        out = self.traffic(x[:, config.TRAFFIC_COL_INDEX[0]])
        s_out = self.search(x[:, config.SEARCH_COL_INDEX])

        out = torch.cat([out, s_out], dim=1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


class Conv2dLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.traffic = TrafficConv2dLSTM()
        self.search = SearchConv2d()

        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()

    def forward(self, x):
        assert x.dim() == 4, "historical input size should be N x D x T x S"

        out = self.traffic(x[:, config.TRAFFIC_COL_INDEX])
        s_out = self.search(x[:, config.SEARCH_COL_INDEX])

        out = torch.cat([out, s_out], dim=1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


class Embedding_Conv2dLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = CategoricalEmbedding()
        self.traffic = TrafficConv2dLSTM()
        self.search = SearchConv2d()

        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()

    def forward(self, x):
        assert isinstance(
            x, tuple
        ), "input should be tuple of (historical data, static data)"
        assert (
            len(x) == 2
        ), "input should be tuple of (historical data, static data)"
        x_dy, x_st = x

        assert x_dy.dim() == 4, "historical input size should be N x D x T x S"
        assert (
            x_st.dim() == 2
        ), "static input size should be N x STATIC_COL_LENGTH"

        emb = self.embedding(x_st.to(dtype=torch.int64))
        out = self.traffic(x_dy[:, config.TRAFFIC_COL_INDEX])
        s_out = self.search(x_dy[:, config.SEARCH_COL_INDEX])

        out = torch.cat([out, s_out, emb], dim=1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out
