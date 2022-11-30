import torch
import torch.nn as nn

from config import config, hparams


class ContextEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.road_n_category = config.N_SEGMENT
        self.datetime_n_category = config.N_DATETIME_GROUP

        self.road_n_embedding = hparams.ROAD_N_EMBEDDING
        self.datetime_n_embedding = hparams.DATETIME_N_EMBEDDING

        self.road_embedding = nn.Embedding(self.road_n_category, self.road_n_embedding)
        self.datetime_embedding = nn.Embedding(
            self.datetime_n_category, self.datetime_n_embedding
        )

        self.embedding_dropout = nn.Dropout(p=hparams.CONTEXT_EMBEDDING_DROPOUT)

    def forward(self, x_road, x_datetime):
        # x_road: N x 1 x D(=1)
        N, _, _ = x_road.shape
        # x_datetime: N x P_hour x D(=1)
        _, P_hour, _ = x_datetime.shape
        assert x_road.shape[-1] == x_datetime.shape[-1] == 1

        # N x T x D(=1) -> N x T
        x_road = x_road[..., 0]
        x_datetime = x_datetime[..., 0]

        # Embedding: N x T -> N x T x H_emb
        out_road = self.road_embedding(x_road)
        out_road = self.embedding_dropout(out_road)

        out_datetime = self.datetime_embedding(x_datetime)
        out_datetime = self.embedding_dropout(out_datetime)

        # Repeat: N x 1 x H_road -> N x P_hour x H_road
        out_road = out_road.repeat(1, P_hour, 1)

        # Concat: (N x P_hour x H_road, N x P_hour x H_datetime)
        # -> N x P_hour x (H_road + H_datetime)
        out = torch.cat([out_road, out_datetime], dim=-1)
        assert out.shape[-1] == self.road_n_embedding + self.datetime_n_embedding

        return out
