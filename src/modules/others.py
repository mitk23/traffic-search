import torch
import torch.nn as nn
import torch.nn.functional as F

from config import hparams


class FeatureConcatLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_traffic, x_specified, x_unspecified, x_context):
        # x_traffic: N x P_hour x H_traffic
        N, P_hour, H_traffic = x_traffic.shape
        # x_specified: N x P_hour x H_specified
        _, _, H_specified = x_specified.shape
        # x_unspecified: N x P_day x H_unspecified
        _, P_day, H_unspecified = x_unspecified.shape
        # x_context: N x P_hour x H_context
        _, _, H_context = x_context.shape

        H_concat = H_traffic + H_specified + H_unspecified + H_context

        x_unspecified = torch.cat(
            [t.repeat(1, P_hour // P_day, 1) for t in x_unspecified.split(1, dim=1)],
            dim=1,
        )
        # x_unspecified = x_unspecified.repeat(1, P_hour // P_day, 1)

        out = torch.cat([x_traffic, x_specified, x_unspecified, x_context], dim=-1)
        assert (
            (out.shape[0] == N)
            and (out.shape[1] == P_hour)
            and (out.shape[2] == H_concat)
        )

        return out


class AffineLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_dim_list = hparams.FC_DIM_LIST
        self.fc_list = nn.ModuleList(
            [
                nn.Linear(in_dim, out_dim)
                for (in_dim, out_dim) in zip(self.fc_dim_list, self.fc_dim_list[1:])
            ]
        )

    def forward(self, x):
        # N x P x H
        N, P, H = x.shape

        out = x
        for i_fc, fc in enumerate(self.fc_list):
            out = fc(out)
            if i_fc < len(self.fc_list) - 1:
                out = F.relu(out)

        return out
