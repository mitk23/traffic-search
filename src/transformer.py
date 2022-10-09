import pickle

import pandas as pd
import torch

from config import FEATURE_COL, STATIC_COL, TARGET_COL
from scaler import STMatrixStandardScaler
from storage import ROAD_TABLE, SCALER_PATH


def matrix2tensor(df, road_table, feature_col, target_col):
    # データ数, 特徴量数
    N = df.shape[0]
    D = len(feature_col)
    # 区間数
    S = road_table.shape[0]
    # 時系列長
    T = N // S

    # テンソルを準備
    X = torch.empty((D, T, S), dtype=torch.float32)
    y = torch.empty((1, T, S), dtype=torch.float32)

    for sec_id, (s_name, e_name, *_) in road_table.iterrows():
        df_sec = df.query(f'start_name == "{s_name}" & end_name == "{e_name}"')

        data = df_sec.loc[:, feature_col]
        target = df_sec.loc[:, target_col]
        X[..., sec_id] = torch.from_numpy(data.values).permute(1, 0)
        y[0, :, sec_id] = torch.from_numpy(target.values)

    return X, y


def scale(X, skip_features, training, scaler_path=SCALER_PATH):
    """
    Spatial-Temporal Tensorを標準化する
    Scalerは訓練時にscaler_pathに保存される

    Parameters
    ----------
    X: torch.Tensor
    skip_features: List[int] | Tuple[int]
    training: bool
    scaler_path: str
    """
    if training:
        scaler = STMatrixStandardScaler(skip_features=skip_features)
        X_norm = scaler.fit_transform(X)

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
    else:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        X_norm = scaler.transform(X)

    return X_norm


def main(args):
    df = pd.read_pickle(args.input_path)
    rd_table = pd.read_pickle(ROAD_TABLE)
    print(f"convert matrix ({df.shape}) to tensor ...")
    X, y = matrix2tensor(df, rd_table, FEATURE_COL, TARGET_COL)
    print(f"converted to tensor (feature: {X.shape}, label: {y.shape})")

    if args.scale:
        print('standardize tensor...')
        training = not args.valid
        scaler_path = args.scaler_path if args.scaler_path else SCALER_PATH
        X_norm = scale(
            X, STATIC_COL, training=training, scaler_path=scaler_path
        )
        print(f'finished standardization. [scaler] {scaler_path}')
        torch.save(X_norm, args.output_feature_path)
    else:
        torch.save(X, args.output_feature_path)

    torch.save(y, args.output_label_path)
    print(
        f"saved feature to {args.output_feature_path}, label to {args.output_label_path}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="path of dataframe pickle")
    parser.add_argument("output_feature_path", help="path of dataframe pickle")
    parser.add_argument("output_label_path", help="path of dataframe pickle")
    parser.add_argument(
        "-s",
        "--scale",
        help="whether to standardize data",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--valid",
        help="set the scaler to validation mode (default: train)",
        action="store_true",
    )
    parser.add_argument("--scaler_path", help="path to save scaler")
    args = parser.parse_args()

    main(args)
