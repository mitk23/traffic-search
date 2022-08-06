import torch


def cyclic_encode(X):
    eps = 1e-8
    max_v, _ = X.view(-1, X.shape[-1]).max(dim=0)
    X_cos = torch.cos(2 * torch.pi * X) / (max_v + eps)
    X_sin = torch.sin(2 * torch.pi * X) / (max_v + eps)
    return X_cos, X_sin
