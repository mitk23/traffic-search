import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config, hparams


class TrafficSearchEncoder(nn.Module):
    def __init__(
        self,
        conv_dim,
        kernel,
        lstm_dim,
        lstm_layers,
        lstm_dropout,
        bidirectional=True,
    ):
        super().__init__()

        self.lstm_dropout = lstm_dropout
        self.bidirectional = bidirectional

        self.conv = nn.Conv2d(
            1,
            conv_dim,
            kernel,
            padding=(kernel[0] // 2, 0),
            padding_mode="replicate",
        )
        self.lstm = nn.LSTM(
            conv_dim,
            lstm_dim,
            lstm_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x):
        N, T, S = x.shape

        out = F.relu(self.conv(x.unsqueeze(1)))
        # N x C x T -> N x T x C
        out = out[..., 0].permute(0, 2, 1)
        # N x T x C -> N x T x H, (L x N x H, L x N x H)
        outs, (h, c) = self.lstm(out)

        return outs, (h, c)


class SearchUnspecEncoder(nn.Module):
    def __init__(self, p_dropout):
        super().__init__()

        self.p_dropout = p_dropout

        self.conv = nn.Conv1d(
            1,
            hparams.UNSPECIFIED_SEARCH_COUNT_HIDDEN,
            hparams.UNSPECIFIED_SEARCH_COUNT_CONV_SPATIAL_KERNEL,
            padding_mode="replicate",
        )
        # self.dropout = nn.Dropout(p=0)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        N, T, S = x.shape
        out = F.relu(self.conv(x))
        # N x C x T -> N x T x C
        out = out.permute(0, 2, 1)
        out = self.dropout(out)

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        lstm_dropout,
        unspec_search_dropout=0.4,
        include_search=True,
        bidirectional=True,
    ):
        super().__init__()

        self.lstm_dropout = lstm_dropout
        self.unspec_search_dropout = unspec_search_dropout
        self.include_search = include_search
        self.bidirectional = bidirectional

        self.traffic_encoder = TrafficSearchEncoder(
            hparams.TRAFFIC_VOLUME_CONV_DIM,
            hparams.TRAFFIC_VOLUME_CONV_KERNEL,
            hparams.TRAFFIC_VOLUME_LSTM_DIM,
            hparams.TRAFFIC_VOLUME_LSTM_LAYERS,
            lstm_dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        if include_search:
            self.search_encoder = TrafficSearchEncoder(
                hparams.SPECIFIED_SEARCH_COUNT_CONV_DIM,
                hparams.SPECIFIED_SEARCH_COUNT_CONV_KERNEL,
                hparams.SPECIFIED_SEARCH_COUNT_LSTM_DIM,
                hparams.SPECIFIED_SEARCH_COUNT_LSTM_LAYERS,
                lstm_dropout=lstm_dropout,
                bidirectional=bidirectional,
            )
            self.unspec_search_encoder = SearchUnspecEncoder(
                p_dropout=unspec_search_dropout
            )

    def forward(self, x_trf, x_sr, x_un_sr):
        # N x T x S -> N x T x H_t, (bi*L_t x N x H_t, bi*L_t x N x H_t)
        outs_trf, state_trf = self.traffic_encoder(x_trf)
        if self.bidirectional:
            # 2*L_t x N x H_t -> L_t x N x 2*H_t
            L2, N, H_t = state_trf[0].shape
            h = (
                state_trf[0]
                .transpose(0, 1)
                .reshape(N, L2 // 2, -1)
                .transpose(0, 1)
                .contiguous()
            )
            c = (
                state_trf[1]
                .transpose(0, 1)
                .reshape(N, L2 // 2, -1)
                .transpose(0, 1)
                .contiguous()
            )
            state_trf = (h, c)

        if self.include_search:
            # N x P x S -> N x P x H_s, (L_s x N x H_s, L_s x N x H_s)
            outs_sr, state_sr = self.search_encoder(x_sr)
            # N x 1 x S -> N x 1 x H_u
            out_un_sr = self.unspec_search_encoder(x_un_sr)
            return (outs_trf, state_trf), outs_sr, out_un_sr
        return outs_trf, state_trf


class CategoricalEmbedding(nn.Module):
    def __init__(self, category_size, emb_size):
        super().__init__()

        self.category_size = category_size
        self.emb_size = emb_size
        self.embedding = nn.Embedding(category_size, emb_size)

    def forward(self, x):
        out = self.embedding(x.to(dtype=torch.int64))
        return out


class TrafficDecoder(nn.Module):
    def __init__(self, lstm_dropout, bidirectional=True):
        super().__init__()

        self.lstm_dropout = lstm_dropout

        self.hid_dim = (
            2 * hparams.TRAFFIC_VOLUME_LSTM_DIM
            if bidirectional
            else hparams.TRAFFIC_VOLUME_LSTM_DIM
        )

        self.bnorm = nn.BatchNorm1d(1)
        self.lstm = nn.LSTM(
            1,
            self.hid_dim,
            hparams.TRAFFIC_VOLUME_LSTM_LAYERS,
            dropout=lstm_dropout,
            batch_first=True,
        )

    def forward(self, x, state):
        N, _, P = x.shape
        # normalize
        x = self.bnorm(x)

        # N x C=1 x P -> N x P x C=1
        x = x.permute(0, 2, 1)
        # N x P x C, (L x N x H_t, L x N x H_t)
        # -> N x P x H_t, (L x N x H_t, L x N x H_t)
        outs, state = self.lstm(x, state)
        return outs, state


class AffineDecoder(nn.Module):
    def __init__(self, embedding_dropout, include_search=True, bidirectional=True):
        super().__init__()

        self.embedding_dropout = embedding_dropout
        self.include_search = include_search
        self.bidirectional = bidirectional

        if include_search:
            if bidirectional:
                self.n_dim = (
                    2 * hparams.TRAFFIC_VOLUME_LSTM_DIM
                    + 2 * hparams.SPECIFIED_SEARCH_COUNT_LSTM_DIM
                    + hparams.UNSPECIFIED_SEARCH_COUNT_HIDDEN
                    + hparams.DATETIME_N_EMBEDDING
                    + hparams.ROAD_N_EMBEDDING
                )
            else:
                self.n_dim = (
                    hparams.TRAFFIC_VOLUME_LSTM_DIM
                    + hparams.SPECIFIED_SEARCH_COUNT_LSTM_DIM
                    + hparams.UNSPECIFIED_SEARCH_COUNT_HIDDEN
                    + hparams.DATETIME_N_EMBEDDING
                    + hparams.ROAD_N_EMBEDDING
                )
        else:
            if bidirectional:
                self.n_dim = (
                    2 * hparams.TRAFFIC_VOLUME_LSTM_DIM
                    + hparams.DATETIME_N_EMBEDDING
                    + hparams.ROAD_N_EMBEDDING
                )
            else:
                self.n_dim = (
                    hparams.TRAFFIC_VOLUME_LSTM_DIM
                    + hparams.DATETIME_N_EMBEDDING
                    + hparams.ROAD_N_EMBEDDING
                )

        self.datetime_embedding = CategoricalEmbedding(
            config.DT_TABLE_SIZE, hparams.DATETIME_N_EMBEDDING
        )
        self.road_embedding = CategoricalEmbedding(
            config.SEC_TABLE_SIZE, hparams.ROAD_N_EMBEDDING
        )
        self.emb_dropout = nn.Dropout(p=embedding_dropout)

        self.fc1 = nn.Linear(self.n_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, trf_dec, sr_enc, un_sr_enc, dt, rd):
        # traffic_dec: N x P x H_t
        N, P, H_t = trf_dec.shape

        # N x P -> N x P x H_d
        dt_emb = self.datetime_embedding(dt)
        dt_emb = self.emb_dropout(dt_emb)
        # N x 1 -> N x 1 x H_r
        rd_emb = self.road_embedding(rd)
        rd_emb = self.emb_dropout(rd_emb)

        if self.include_search:
            outs = torch.cat(
                [
                    dt_emb,
                    rd_emb.repeat(1, P, 1),
                    sr_enc,
                    un_sr_enc.repeat(1, P, 1),
                    trf_dec,
                ],
                dim=-1,
            )
        else:
            outs = torch.cat([dt_emb, rd_emb.repeat(1, P, 1), trf_dec], dim=-1)
        # N x P x SUM -> N x P x H_fc
        outs = F.relu(self.fc1(outs))
        # N x P x H_fc -> N x P x 1
        outs = self.fc2(outs)

        return outs


class Decoder(nn.Module):
    def __init__(
        self,
        lstm_dropout,
        embedding_dropout=0.4,
        include_search=True,
        bidirectional=True,
    ):
        super().__init__()

        self.lstm_dropout = lstm_dropout
        self.embedding_dropout = embedding_dropout
        self.include_search = include_search

        self.traffic_decoder = TrafficDecoder(
            lstm_dropout=lstm_dropout, bidirectional=bidirectional
        )
        self.affine_decoder = AffineDecoder(
            embedding_dropout=embedding_dropout,
            include_search=include_search,
            bidirectional=bidirectional,
        )

    def forward(self, x, trf_enc, sr_enc, un_sr_enc, dt, rd):
        # N x 1 x P, (L x N x H_t, L x N x H_t)
        # -> N x P x H_t, (L x N x H_t, L x N x H_t)
        outs_trf, state_trf = self.traffic_decoder(x, trf_enc)
        outs = self.affine_decoder(outs_trf, sr_enc, un_sr_enc, dt, rd)
        # N x P x 1 -> N x P
        outs = outs[..., 0]
        return outs

    def generate(self, trf_enc, sr_enc, un_sr_enc, dt, rd, start_value=-1.0):
        with torch.no_grad():
            # N x 1 x 1
            N = trf_enc[0].shape[1]
            out = torch.tensor(start_value).repeat(N).unsqueeze(-1).unsqueeze(-1)
            out = out.to(trf_enc[0].device)

            state = trf_enc

            generated = []

            for i in range(hparams.PREDICTION_HORIZON):
                out, state = self.traffic_decoder(out, state)
                if self.include_search:
                    out = self.affine_decoder(
                        out, sr_enc[:, [i]], un_sr_enc, dt[:, [i]], rd
                    )
                else:
                    out = self.affine_decoder(out, None, None, dt[:, [i]], rd)

                generated.append(out)

        # N x P x 1
        generated = torch.cat(generated, dim=1)
        # N x P x 1 -> N x P
        generated = generated[..., 0]
        return generated


class LSTMEncoder(nn.Module):
    def __init__(self, lstm_dropout, bidirectional=True):
        super().__init__()

        self.lstm_dropout = lstm_dropout
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            1,
            hparams.TRAFFIC_VOLUME_LSTM_DIM,
            hparams.TRAFFIC_VOLUME_LSTM_LAYERS,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
            batch_first=True,
        )

    def forward(self, x):
        N, T, S = x.shape
        x = x[..., [S // 2]]
        # N x T x 1 -> N x T x H, (L x N x H, L x N x H)
        outs, (h, c) = self.lstm(x)

        if self.bidirectional:
            # 2*L_t x N x H_t -> L_t x N x 2*H_t
            L2, N, H_t = h.shape
            h = h.transpose(0, 1).reshape(N, L2 // 2, -1).transpose(0, 1).contiguous()
            c = h.transpose(0, 1).reshape(N, L2 // 2, -1).transpose(0, 1).contiguous()

        return outs, (h, c)


class CNNLSTMEncoder(nn.Module):
    def __init__(self, lstm_dropout, bidirectional=True):
        super().__init__()

        self.lstm_dropout = lstm_dropout
        self.bidirectional = bidirectional

        self.conv = nn.Conv2d(
            1,
            hparams.TRAFFIC_VOLUME_CONV_DIM,
            hparams.TRAFFIC_VOLUME_CONV_KERNEL,
            padding=(hparams.TRAFFIC_VOLUME_CONV_KERNEL[0] // 2, 0),
            padding_mode="replicate",
        )
        self.lstm = nn.LSTM(
            hparams.TRAFFIC_VOLUME_CONV_DIM,
            hparams.TRAFFIC_VOLUME_LSTM_DIM,
            hparams.TRAFFIC_VOLUME_LSTM_LAYERS,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x):
        N, T, S = x.shape

        out = F.relu(self.conv(x.unsqueeze(1)))
        # N x C x T -> N x T x C
        out = out[..., 0].permute(0, 2, 1)
        # N x T x C -> N x T x H, (L x N x H, L x N x H)
        outs, (h, c) = self.lstm(out)

        if self.bidirectional:
            # 2*L_t x N x H_t -> L_t x N x 2*H_t
            L2, N, H_t = h.shape
            h = h.transpose(0, 1).reshape(N, L2 // 2, -1).transpose(0, 1).contiguous()
            c = h.transpose(0, 1).reshape(N, L2 // 2, -1).transpose(0, 1).contiguous()

        return outs, (h, c)


class LSTMDecoder(nn.Module):
    def __init__(self, lstm_dropout, bidirectional=True):
        super().__init__()

        self.lstm_dropout = lstm_dropout

        self.hid_dim = (
            2 * hparams.TRAFFIC_VOLUME_LSTM_DIM
            if bidirectional
            else hparams.TRAFFIC_VOLUME_LSTM_DIM
        )

        self.bnorm = nn.BatchNorm1d(1)
        self.lstm = nn.LSTM(
            1,
            self.hid_dim,
            hparams.TRAFFIC_VOLUME_LSTM_LAYERS,
            dropout=lstm_dropout,
            batch_first=True,
        )

        self.fc1 = nn.Linear(self.hid_dim, hparams.FC_EMB)
        self.fc2 = nn.Linear(hparams.FC_EMB, 1)

    def forward(self, x, state):
        N, _, P = x.shape
        # normalize
        x = self.bnorm(x)

        # N x C=1 x P -> N x P x C=1
        x = x.permute(0, 2, 1)
        # N x P x C, (L x N x H_t, L x N x H_t)
        # -> N x P x H_t, (L x N x H_t, L x N x H_t)
        outs, state = self.lstm(x, state)
        outs = F.relu(self.fc1(outs))
        outs = self.fc2(outs)

        return outs, state

    def generate(self, trf_enc, start_value=-1.0):
        with torch.no_grad():
            # N x 1 x 1
            N = trf_enc[0].shape[1]
            out = torch.tensor(start_value).repeat(N).unsqueeze(-1).unsqueeze(-1)
            out = out.to(trf_enc[0].device)

            state = trf_enc

            generated = []

            for i in range(24):
                out, state = self.forward(out, state)
                generated.append(out)

        # N x P x 1
        generated = torch.cat(generated, dim=1)
        # N x P x 1 -> N x P
        generated = generated[..., 0]
        return generated


class LSTMEmbeddingDecoder(nn.Module):
    def __init__(self, lstm_dropout, embedding_dropout=0.4, bidirectional=True):
        super().__init__()

        self.lstm_dropout = lstm_dropout
        self.embedding_dropout = embedding_dropout

        self.hid_dim = (
            2 * hparams.TRAFFIC_VOLUME_LSTM_DIM
            if bidirectional
            else hparams.TRAFFIC_VOLUME_LSTM_DIM
        )

        self.datetime_embedding = CategoricalEmbedding(
            config.DT_TABLE_SIZE, hparams.DATETIME_N_EMBEDDING
        )
        self.road_embedding = CategoricalEmbedding(
            config.SEC_TABLE_SIZE, hparams.ROAD_N_EMBEDDING
        )
        self.emb_dropout = nn.Dropout(p=embedding_dropout)

        self.bnorm = nn.BatchNorm1d(1)
        self.lstm = nn.LSTM(
            1,
            self.hid_dim,
            hparams.TRAFFIC_VOLUME_LSTM_LAYERS,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.fc1 = nn.Linear(
            self.hid_dim + hparams.DATETIME_N_EMBEDDING + hparams.ROAD_N_EMBEDDING,
            hparams.FC_EMB,
        )
        self.fc2 = nn.Linear(hparams.FC_EMB, 1)

    def forward(self, x, state, dt, rd):
        N, _, P = x.shape
        # normalize
        x = self.bnorm(x)

        # N x C=1 x P -> N x P x C=1
        x = x.permute(0, 2, 1)
        # N x P x C, (L x N x H_t, L x N x H_t)
        # -> N x P x H_t, (L x N x H_t, L x N x H_t)
        outs, state = self.lstm(x, state)

        # N x P -> N x P x H_d
        dt_emb = self.datetime_embedding(dt)
        dt_emb = self.emb_dropout(dt_emb)
        # N x 1 -> N x 1 x H_r
        rd_emb = self.road_embedding(rd)
        rd_emb = self.emb_dropout(rd_emb)

        outs = torch.cat([dt_emb, rd_emb.repeat(1, P, 1), outs], dim=-1)

        outs = F.relu(self.fc1(outs))
        outs = self.fc2(outs)

        return outs, state

    def generate(self, trf_enc, dt, rd, start_value=-1.0):
        with torch.no_grad():
            # N x 1 x 1
            N = trf_enc[0].shape[1]
            out = torch.tensor(start_value).repeat(N).unsqueeze(-1).unsqueeze(-1)
            out = out.to(trf_enc[0].device)

            state = trf_enc

            generated = []

            for i in range(24):
                out, state = self.forward(out, state, dt[:, [i]], rd)
                generated.append(out)

        # N x P x 1
        generated = torch.cat(generated, dim=1)
        # N x P x 1 -> N x P
        generated = generated[..., 0]
        return generated
