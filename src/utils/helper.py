import sys

sys.path.append("..")
import random

import config
import numpy as np
import torch


def format_stmatrix(df, sec_table, feature_col, target_col):
    # データ数, 特徴量数
    N = df.shape[0]
    D = len(feature_col)
    # 区間数
    S = sec_table.shape[0]
    # 時系列長
    T = N // S

    # テンソルを準備
    # X = torch.empty((S, T, D), dtype=torch.float32)
    # y = torch.empty((S, T, 1), dtype=torch.float32)
    X = torch.empty((D, T, S), dtype=torch.float32)
    y = torch.empty((1, T, S), dtype=torch.float32)

    for sec_id, (s_name, e_name, *_) in sec_table.iterrows():
        query = f'start_name == "{s_name}" & end_name == "{e_name}"'
        df_sec = df.query(query)

        data = df_sec.loc[:, feature_col]
        target = df_sec.loc[:, target_col]
        X[..., sec_id] = torch.from_numpy(data.values).permute(1, 0)
        y[0, :, sec_id] = torch.from_numpy(target.values)

    return X, y


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


def fix_seed(seed=config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


def seed_worker(worker_id):
    '''
    dataloaderのseedを固定する
    '''
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



