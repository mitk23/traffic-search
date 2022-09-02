import torch
import torch.nn as nn
import torch.nn.functional as F

import config

TRAFFIC_CONV = 64
TRAFFIC_HIDDEN = 128
TRAFFIC_LSTM_LAYERS = 2
TRAFFIC_KERNEL = (7, 5)

SEARCH_CONV = 64
SEARCH_HIDDEN = 128
SEARCH_LSTM_LAYERS = 2
SEARCH_KERNEL = (7, 5)

UNSPEC_SEARCH_HIDDEN = 64
UNSPEC_SEARCH_KERNEL = 5

DATETIME_EMB = 32
ROAD_EMB = 16

FC_EMB = 32


class TrafficSearchEncoder(nn.Module):
    def __init__(self, conv_dim, kernel, lstm_dim, lstm_layers):
        super().__init__()

        self.conv = nn.Conv2d(
            1,
            conv_dim,
            kernel,
            padding=(kernel[0] // 2, 0),
            padding_mode="replicate",
        )
        self.lstm = nn.LSTM(conv_dim, lstm_dim, lstm_layers, batch_first=True)

    def forward(self, x):
        N, T, S = x.shape

        out = F.relu(self.conv(x.unsqueeze(1)))
        # N x C x T -> N x T x C
        out = out[..., 0].permute(0, 2, 1)
        # N x T x C -> N x T x H_t, (L x N x H_t, L x N x H_t)
        outs, (h, c) = self.lstm(out)

        return outs, (h, c)


class SearchUnspecEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv1d(
            1,
            UNSPEC_SEARCH_HIDDEN,
            UNSPEC_SEARCH_KERNEL,
            padding_mode="replicate",
        )

    def forward(self, x):
        N, T, S = x.shape
        out = F.relu(self.conv(x))
        # N x C x T -> N x T x C
        out = out.permute(0, 2, 1)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.traffic_encoder = TrafficSearchEncoder(
            TRAFFIC_CONV, TRAFFIC_KERNEL, TRAFFIC_HIDDEN, TRAFFIC_LSTM_LAYERS
        )
        self.search_encoder = TrafficSearchEncoder(
            SEARCH_CONV, SEARCH_KERNEL, SEARCH_HIDDEN, SEARCH_LSTM_LAYERS
        )
        self.unspec_search_encoder = SearchUnspecEncoder()

    def forward(self, x_trf, x_sr, x_un_sr):
        # N x T x S -> N x T x H_t, (L_t x N x H_t, L_t x N x H_t)
        outs_trf, state_trf = self.traffic_encoder(x_trf)
        # N x P x S -> N x P x H_s, (L_s x N x H_s, L_s x N x H_s)
        outs_sr, state_sr = self.search_encoder(x_sr)
        # N x 1 x S -> N x P x H_u
        out_un_sr = self.unspec_search_encoder(x_un_sr)

        return (outs_trf, state_trf), outs_sr, out_un_sr


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
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            1, TRAFFIC_HIDDEN, TRAFFIC_LSTM_LAYERS, batch_first=True
        )

    def forward(self, x, encoded):
        N, _, P = x.shape
        h, c = encoded
        state = (h[0].repeat(2, 1, 1), c[0].repeat(2, 1, 1))

        # N x C=1 x P -> N x P x C=1
        x = x.permute(0, 2, 1)
        # N x P x C, (L x N x H_t, L x N x H_t) -> N x P x H_t, (L x N x H_t, L x N x H_t)
        outs, state = self.lstm(x, state)
        return outs, state


class AffineDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_dim = (
            TRAFFIC_HIDDEN
            + SEARCH_HIDDEN
            + UNSPEC_SEARCH_HIDDEN
            + DATETIME_EMB
            + ROAD_EMB
        )

        self.datetime_embedding = CategoricalEmbedding(
            config.DT_TABLE_SIZE, DATETIME_EMB
        )
        self.road_embedding = CategoricalEmbedding(
            config.SEC_TABLE_SIZE, ROAD_EMB
        )

        self.fc1 = nn.Linear(self.n_dim, FC_EMB)
        self.fc2 = nn.Linear(FC_EMB, 1)

    def forward(self, trf_dec, sr_enc, un_sr_enc, dt, rd):
        # traffic_dec: N x P x H_t
        N, P, H_t = trf_dec.shape
        # search_enc: N x P x H_s
        _, _, H_s = sr_enc.shape
        # un_search_enc: N x 1 x H_u
        _, _, H_u = un_sr_enc.shape

        # N x P -> N x P x H_d
        dt_emb = self.datetime_embedding(dt)
        # N x 1 -> N x 1 x H_r
        rd_emb = self.road_embedding(rd)

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
        # N x P x SUM -> N x P x H_fc
        outs = F.relu(self.fc1(outs))
        # N x P x H_fc -> N x P x 1
        outs = self.fc2(outs)

        return outs


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.traffic_decoder = TrafficDecoder()
        self.affine_decoder = AffineDecoder()

    def forward(self, x, trf_enc, sr_enc, un_sr_enc, dt, rd):
        # N x 1 x P, (L x N x H_t, L x N x H_t) -> N x P x H_t, (L x N x H_t, L x N x H_t)
        outs_trf, state_trf = self.traffic_decoder(x, trf_enc)
        outs = self.affine_decoder(outs_trf, sr_enc, un_sr_enc, dt, rd)
        # N x P x 1 -> N x P
        outs = outs[..., 0]
        return outs

    def generate(self, trf_enc, sr_enc, un_sr_enc, dt, rd, start_value=-1.0):
        with torch.no_grad():
            # N x 1 x 1
            N = sr_enc.shape[0]
            out = (
                torch.tensor(start_value).repeat(N).unsqueeze(-1).unsqueeze(-1)
            )
            out = out.to(sr_enc.device)
            generated = []

            for i in range(24):
                x = out
                out_trf, state_trf = self.traffic_decoder(x, trf_enc)
                out = self.affine_decoder(
                    out_trf, sr_enc[:, [i]], un_sr_enc, dt[:, [i]], rd
                )

                generated.append(out)

        # N x P x 1
        generated = torch.cat(generated, dim=1)
        # N x P x 1 -> N x P
        generated = generated[..., 0]
        return generated


class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, features, decoder_xs):
        dt, rd, sr, un_sr, trf = features

        (outs_trf, state_trf), outs_sr, outs_un_sr = self.encoder(
            trf, sr, un_sr
        )
        outs = self.decoder(decoder_xs, state_trf, outs_sr, outs_un_sr, dt, rd)

        return outs

    def generate(self, features, start_value=-1.0):
        dt, rd, sr, un_sr, trf = features

        (outs_trf, state_trf), outs_sr, outs_un_sr = self.encoder(
            trf, sr, un_sr
        )
        generated = self.decoder.generate(
            state_trf, outs_sr, outs_un_sr, dt, rd, start_value
        )
        return generated


class WithoutEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.traffic_encoder = TrafficSearchEncoder(TRAFFIC_CONV, TRAFFIC_KERNEL, TRAFFIC_HIDDEN, TRAFFIC_LSTM_LAYERS)
        
    def forward(self, x_trf):
        # N x T x S -> N x T x H_t, (L_t x N x H_t, L_t x N x H_t)
        outs_trf, state_trf = self.traffic_encoder(x_trf)
        return outs_trf, state_trf


class WithoutAffineDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.n_dim = TRAFFIC_HIDDEN + DATETIME_EMB + ROAD_EMB
        
        self.datetime_embedding = CategoricalEmbedding(config.DT_TABLE_SIZE, DATETIME_EMB)
        self.road_embedding = CategoricalEmbedding(config.SEC_TABLE_SIZE, ROAD_EMB)
        
        self.fc1 = nn.Linear(self.n_dim, FC_EMB)
        self.fc2 = nn.Linear(FC_EMB, 1)
        
    def forward(self, trf_dec, dt, rd):
        # traffic_dec: N x P x H_t
        N, P, H_t = trf_dec.shape
        
        # N x P -> N x P x H_d
        dt_emb = self.datetime_embedding(dt)
        # N x 1 -> N x 1 x H_r
        rd_emb = self.road_embedding(rd)
        
        outs = torch.cat([
            dt_emb, 
            rd_emb.repeat(1, P, 1), 
            trf_dec
        ], dim=-1)
        # N x P x SUM -> N x P x H_fc
        outs = F.relu(self.fc1(outs))
        # N x P x H_fc -> N x P x 1
        outs = self.fc2(outs)
        
        return outs 


class WithoutDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.traffic_decoder = TrafficDecoder()
        self.affine_decoder = WithoutAffineDecoder()
    
    def forward(self, x, trf_enc, dt, rd):
        # N x 1 x P, (L x N x H_t, L x N x H_t) -> N x P x H_t, (L x N x H_t, L x N x H_t)
        outs_trf, state_trf = self.traffic_decoder(x, trf_enc)
        outs = self.affine_decoder(outs_trf, dt, rd)
        # N x P x 1 -> N x P
        outs = outs[..., 0]
        return outs
    
    def generate(self, trf_enc, dt, rd, start_value=-1.0):
        with torch.no_grad():
            # N x 1 x 1
            N = dt.shape[0]
            out = torch.tensor(start_value).repeat(N).unsqueeze(-1).unsqueeze(-1)
            out = out.to(dt.device)
            generated = []

            for i in range(24):
                x = out
                out_trf, state_trf = self.traffic_decoder(x, trf_enc)
                out = self.affine_decoder(out_trf, dt[:, [i]], rd)

                generated.append(out)
        
        # N x P x 1
        generated = torch.cat(generated, dim=1)
        # N x P x 1 -> N x P
        generated = generated[..., 0]
        return generated


class WithoutEncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = WithoutEncoder()
        self.decoder = WithoutDecoder()
        
    def forward(self, features, decoder_xs):
        dt, rd, sr, un_sr, trf = features
        
        outs_trf, state_trf= self.encoder(trf)
        outs = self.decoder(decoder_xs, state_trf, dt, rd)

        return outs
    
    def generate(self, features, start_value=-1.0):
        dt, rd, sr, un_sr, trf = features
        
        outs_trf, state_trf = self.encoder(trf)
        generated = self.decoder.generate(state_trf, dt, rd, start_value)
        return generated
