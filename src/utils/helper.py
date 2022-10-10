import sys

sys.path.append("..")
import random

import numpy as np
import pandas as pd
import torch
from hparams import RANDOM_SEED


def clip_period(df, start, end):
    """
    [start, end)の期間を抽出

    Parameters
    ----------
    start: str
    end: str
    """
    return df.loc[
        (df["datetime"] >= pd.Timestamp(start))
        & (df["datetime"] < pd.Timestamp(end))
    ].copy()


def cyclic_encode(X):
    eps = 1e-8
    max_v, _ = X.view(-1, X.shape[-1]).max(dim=0)
    X_cos = torch.cos(2 * torch.pi * X) / (max_v + eps)
    X_sin = torch.sin(2 * torch.pi * X) / (max_v + eps)
    return X_cos, X_sin


def train_test_split(X, y, test_ratio):
    assert (
        X.dim() == 3
    ), "X should be Spatial-Temporal Matrix (Features x Periods x Sections)"
    _, T, _ = X.shape
    index_split = int(T * test_ratio)
    X_train, X_test = X[:, :-index_split], X[:, -index_split:]
    y_train, y_test = y[:, :-index_split], y[:, -index_split:]

    return X_train, X_test, y_train, y_test


def fix_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


def seed_worker(worker_id):
    """
    dataloaderのseedを固定する
    """
    worker_seed = RANDOM_SEED % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def validate(predicted, target):
    mae_mat = np.abs(predicted - target)
    mae = mae_mat.mean()

    mse_mat = (predicted - target) ** 2
    rmse = np.sqrt(mse_mat.mean())

    return mae, rmse


def multistep_validate(predicted, target, steps=[1, 3, 6, 12, 18, 24]):
    result = {}

    for step in steps:
        t = step - 1
        mae, rmse = validate(predicted[:, t], target[:, t])
        # print(f'{step} hour ahead: MAE = {mae:.3f}, RMSE = {rmse:.3f}')

        result[f"{step}_ahead"] = {"mae": mae, "rmse": rmse}

    mae, rmse = validate(predicted, target)
    # print(f'Whole: MAE = {mae:.3f}, RMSE = {rmse:.3f}')

    result["whole"] = {"mae": mae, "rmse": rmse}
    return result
