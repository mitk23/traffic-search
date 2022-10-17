import torch.nn as nn

from modules import (
    CNNLSTMEncoder,
    Decoder,
    Encoder,
    LSTMDecoder,
    LSTMEmbeddingDecoder,
    LSTMEncoder,
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
    def __init__(
        self, lstm_dropout=0, include_search=True, bidirectional=True
    ):
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
            (outs_trf, state_trf), outs_sr, outs_un_sr = self.encoder(
                trf, sr, un_sr
            )
        else:
            outs_trf, state_trf = self.encoder(trf, sr, un_sr)
            outs_sr, outs_un_sr = None, None

        outs = self.decoder(decoder_xs, state_trf, outs_sr, outs_un_sr, dt, rd)

        return outs

    def generate(self, features, start_value=-1.0):
        dt, rd, sr, un_sr, trf = features

        if self.include_search:
            (outs_trf, state_trf), outs_sr, outs_un_sr = self.encoder(
                trf, sr, un_sr
            )
        else:
            outs_trf, state_trf = self.encoder(trf, sr, un_sr)
            outs_sr, outs_un_sr = None, None

        generated = self.decoder.generate(
            state_trf, outs_sr, outs_un_sr, dt, rd, start_value
        )
        return generated
