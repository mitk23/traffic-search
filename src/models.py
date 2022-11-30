import torch
import torch.nn as nn

from config import hparams
from module import (
    CNNLSTMEncoder,
    Decoder,
    Encoder,
    LSTMDecoder,
    LSTMEmbeddingDecoder,
    LSTMEncoder,
)
from modules.context_embedding import ContextEmbedding
from modules.others import AffineLayer, FeatureConcatLayer
from modules.search_count_encoder import SearchCountEncoder, TemporalSearchCountEncoder
from modules.traffic_volume_decode_layer import TrafficVolumeDecodeLayer
from modules.traffic_volume_encoder import (
    TemporalTrafficVolumeEncoder,
    TrafficVolumeEncoder,
)


class T_ED(nn.Module):
    def __init__(self, lstm_dropout=0, bidirectional=True):
        super().__init__()

        self.lstm_dropout = lstm_dropout

        self.encoder = LSTMEncoder(
            lstm_dropout=lstm_dropout, bidirectional=bidirectional
        )
        self.decoder = LSTMDecoder(
            lstm_dropout=lstm_dropout, bidirectional=bidirectional
        )

    def forward(self, features, decoder_xs):
        dt, rd, sr, un_sr, trf = features

        outs_trf, state_trf = self.encoder(trf)
        outs, _ = self.decoder(decoder_xs, state_trf)
        outs = outs[..., 0]
        return outs

    def generate(self, features, start_value=-1.0):
        dt, rd, sr, un_sr, trf = features

        outs_trf, state_trf = self.encoder(trf)
        generated = self.decoder.generate(state_trf, start_value)
        return generated


class ST_ED(T_ED):
    def __init__(self, lstm_dropout=0, bidirectional=True):
        super().__init__()

        self.lstm_dropout = lstm_dropout

        self.encoder = CNNLSTMEncoder(
            lstm_dropout=lstm_dropout, bidirectional=bidirectional
        )
        self.decoder = LSTMDecoder(
            lstm_dropout=lstm_dropout, bidirectional=bidirectional
        )


class TE_ED(nn.Module):
    def __init__(self, lstm_dropout=0, bidirectional=True):
        super().__init__()

        self.lstm_dropout = lstm_dropout

        self.encoder = LSTMEncoder(
            lstm_dropout=lstm_dropout, bidirectional=bidirectional
        )
        self.decoder = LSTMEmbeddingDecoder(
            lstm_dropout=lstm_dropout, bidirectional=bidirectional
        )

    def forward(self, features, decoder_xs):
        dt, rd, sr, un_sr, trf = features

        outs_trf, state_trf = self.encoder(trf)
        outs, _ = self.decoder(decoder_xs, state_trf, dt, rd)
        outs = outs[..., 0]
        return outs

    def generate(self, features, start_value=-1.0):
        dt, rd, sr, un_sr, trf = features

        outs_trf, state_trf = self.encoder(trf)
        generated = self.decoder.generate(state_trf, dt, rd, start_value)
        return generated


class STE_ED(nn.Module):
    def __init__(self, lstm_dropout=0, include_search=True, bidirectional=True):
        super().__init__()

        self.lstm_dropout = lstm_dropout
        self.include_search = include_search

        self.encoder = Encoder(
            lstm_dropout=lstm_dropout,
            include_search=include_search,
            bidirectional=bidirectional,
        )
        self.decoder = Decoder(
            lstm_dropout=lstm_dropout,
            include_search=include_search,
            bidirectional=bidirectional,
        )

    def forward(self, features, decoder_xs):
        dt, rd, sr, un_sr, trf = features

        if self.include_search:
            (outs_trf, state_trf), outs_sr, outs_un_sr = self.encoder(trf, sr, un_sr)
        else:
            outs_trf, state_trf = self.encoder(trf, sr, un_sr)
            outs_sr, outs_un_sr = None, None

        outs = self.decoder(decoder_xs, state_trf, outs_sr, outs_un_sr, dt, rd)

        return outs

    def generate(self, features, start_value=-1.0):
        dt, rd, sr, un_sr, trf = features

        if self.include_search:
            (outs_trf, state_trf), outs_sr, outs_un_sr = self.encoder(trf, sr, un_sr)
        else:
            outs_trf, state_trf = self.encoder(trf, sr, un_sr)
            outs_sr, outs_un_sr = None, None

        generated = self.decoder.generate(
            state_trf, outs_sr, outs_un_sr, dt, rd, start_value
        )
        return generated


class STE_ED_S(nn.Module):
    def __init__(self):
        super().__init__()

        self.traffic_volume_encoder = TrafficVolumeEncoder()
        self.search_count_encoder = SearchCountEncoder()
        self.context_embedding = ContextEmbedding()

        self.traffic_volume_decode_layer = TrafficVolumeDecodeLayer()
        self.feature_concat_layer = FeatureConcatLayer()
        self.affine_layer = AffineLayer()

    def forward(self, feature_dict, decoder_x):
        # Traffic Volume Encoder + Search Count Encoder + Context Embedding
        (
            out_traffic,
            state_traffic,
            out_specified,
            out_unspecified,
            out_context,
        ) = self.encode(feature_dict)
        encoder_result = {
            "traffic": {
                "state": state_traffic,
            },
            "specified_search": {
                "out": out_specified,
            },
            "unspecified_search": {
                "out": out_unspecified,
            },
            "context": {
                "out": out_context,
            },
        }
        # Traffic Volume Decode Layer -> Feature Concat Layer -> Affine Layer
        out = self.decode(decoder_x, encoder_result, return_traffic_state=False)

        return out

    def generate(
        self,
        feature_dict,
        start_value=-1.0,
        prediction_horizon=hparams.PREDICTION_HORIZON,
    ):
        with torch.no_grad():
            (
                out_traffic,
                state_traffic,
                out_specified,
                out_unspecified,
                out_context,
            ) = self.encode(feature_dict)

            # N x T x H_traffic
            N, _, _ = out_traffic.shape

            # decoder input: N x P_hour(=1) x D(=1)
            decoder_x = torch.full((N, 1, 1), fill_value=start_value)
            # mapping device
            out = decoder_x.to(out_traffic.device)

            result = []
            for i in range(hparams.PREDICTION_HORIZON):
                # extract specific time step feature for future hourly data
                # (specified_search, context)
                encoder_result = {
                    "traffic": {
                        "state": state_traffic,
                    },
                    "specified_search": {
                        "out": out_specified[:, [i]],
                    },
                    "unspecified_search": {
                        "out": out_unspecified[:, [i // 24]],
                    },
                    "context": {"out": out_context[:, [i]]},
                }

                out, state_traffic = self.decode(
                    out, encoder_result, return_traffic_state=True
                )
                result.append(out)

            # List[N x 1 x 1] -> N x P_hour x 1
            result = torch.cat(result, dim=1)
            return result

    def encode(self, feature_dict):
        # N x T x W x D(=1)
        # -> (N x T x H_traffic, (n_layers x N x H_traffic, n_layers x N x H_traffic))
        out_traffic, state_traffic = self.traffic_volume_encoder(
            feature_dict["traffic"]
        )

        # (N x P_hour x W x D(=1), N x P_day x W x D(=1))
        # -> (N x P_hour x H_specified, N x P_day x H_unspecified)
        out_specified, out_unspecified = self.search_count_encoder(
            feature_dict["specified_search"], feature_dict["unspecified_search"]
        )

        # (N x 1 x 1, N x P_hour x 1) -> N x P_hour x H_context
        out_context = self.context_embedding(
            feature_dict["road"], feature_dict["datetime"]
        )

        return out_traffic, state_traffic, out_specified, out_unspecified, out_context

    def decode(self, decoder_x, encoder_result, return_traffic_state=False):
        # (N x P_hour x D(=1), (n_layers x N x H_traffic, n_layers x N x H_traffic))
        # -> (N x P_hour x H_traffic, (n_layers x N x H_traffic, n_layers x N x H_traffic))
        out_traffic, state_traffic = self.traffic_volume_decode_layer(
            decoder_x, encoder_result["traffic"]["state"]
        )

        # (N x P_hour x H_traffic, N x P_hour x H_specified, N x P_day x H_unspecified, N x P_hour x H_context)
        # -> N x P_hour x H_all
        out = self.feature_concat_layer(
            out_traffic,
            encoder_result["specified_search"]["out"],
            encoder_result["unspecified_search"]["out"],
            encoder_result["context"]["out"],
        )

        # N x P_hour x H_all -> N x P_hour x 1
        out = self.affine_layer(out)

        if return_traffic_state:
            return out, state_traffic
        return out


class TE_ED_S(STE_ED_S):
    def __init__(self):
        super().__init__()

        self.traffic_volume_encoder = TemporalTrafficVolumeEncoder()
        self.search_count_encoder = TemporalSearchCountEncoder()
