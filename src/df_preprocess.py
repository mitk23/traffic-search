"""
- dataframeに対して前処理を行う
    - 期間を限定
    - 列編集
    - 型変換
    - 欠損埋め
- 軽量なpickle形式のデータセットへと変換する
"""
import os
import time

import numpy as np
import pandas as pd

from context_table import save_tables
from storage import COL_TYPES, DATETIME_TABLE, ROAD_TABLE, TRAFFIC_CSV


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


def linear_interpolate(df, col):
    """
    dfのcol列内の欠損を区間ごとに線形補間する

    Parameters
    ----------
    df: pandas.DataFrame
    col: str or List[str]
    """
    l_intp = lambda g: g.interpolate(method="linear", axis=0)

    df.sort_values("datetime", inplace=True)
    df[col] = df.groupby(["start_code", "end_code"])[col].apply(l_intp)
    return df


def edit_columns(df):
    """
    yearを追加, 渋滞量, 方向をbinary化, quarterを数値化, OCC->[0,1]
    """
    df = df.assign(
        year=df["datetime"].dt.year,
        jam_flag=np.where(df["speed"] < 40.0, 1, 0),
        direction=df["direction"].map({"上り": 0, "下り": 1}),
        quarter=df["quarter"].str[-1],
        OCC=df["OCC"] / 100.0,
    )
    return df


def drop_columns(df):
    """
    object型 + 天気列をdrop
    """
    drop_cols = [
        "index",
        "date",
        "road_code",
        "pressure",
        "rainfall",
        "temperature",
        "humidity",
        "wind_speed",
        "daylight_hours",
        "snowfall",
        "deepest_snowfall",
        "weather_description",
        "jam_type",
    ]
    return df.drop(drop_cols, axis=1)


def convert_coltype(df):
    # 型変換
    f32_cols = list(df.select_dtypes(include=[np.float64]).columns)
    i32_cols = list(df.select_dtypes(include=[int]).columns)
    i32_cols += ["start_degree", "end_degree", "degree_sum"]
    cat_cols = {
        "start_name",
        "end_name",
        "start_code",
        "end_code",
        "start_pref_code",
        "end_pref_code",
        "direction",
        "month",
        "day",
        "dayofweek",
        "is_holiday",
        "hour",
        "quarter",
        "jam_flag",
    }

    type_map = {}
    type_map.update({f_col: np.float32 for f_col in f32_cols})
    type_map.update({i_col: np.int32 for i_col in i32_cols})
    type_map.update({c_col: "category" for c_col in cat_cols})

    return df.astype(type_map)


def preprocess(df, start=None, end=None):
    # [start, end]までの期間を抽出
    if start and end:
        df = clip_period(df, start, end)
    # 速度の欠損を埋める
    df = linear_interpolate(df, "speed")
    # 列を削る
    df = drop_columns(df)
    # いくつかの列を編集
    df = edit_columns(df)
    # 型を変換
    df = convert_coltype(df)
    return df


def add_datetime_id(df, dt_table):
    time_col = ["hour", "dayofweek", "is_holiday"]
    f = lambda g: g.assign(datetime_id=dt_table.loc[g.name, "index"])
    df = df.groupby(time_col).apply(f).astype({"datetime_id": "category"})
    return df


def add_road_id(df, sec_table):
    f = lambda g: g.assign(
        section_id=sec_table.query(
            f'start_name == "{g.name[0]}" & end_name == "{g.name[1]}"'
        ).index.item()
    )
    df = (
        df.groupby(["start_name", "end_name"])
        .apply(f)
        .astype({"section_id": "category"})
    )
    return df


def postprocess(df, dt_table, rd_table):
    # コンテキストデータの1次元IDを付与する
    df = add_datetime_id(df, dt_table)
    df = add_road_id(df, rd_table)
    return df


def main(args):
    start = time.time()

    print("start reading...")
    df = pd.read_csv(
        TRAFFIC_CSV, parse_dates=True, index_col="datetime", dtype=COL_TYPES
    ).reset_index()
    print(f"finish reading ({time.time() - start:.2f} [sec])")

    start = time.time()
    print("start preprocessing...")
    df = preprocess(df, start=args.start, end=args.end)
    print(f"finish preprocessing ({time.time() - start:.2f} [sec])")

    df.reset_index(drop=True, inplace=True)
    df.to_pickle(args.save_path)

    print("adding context information to dataframe...")
    if not (os.path.exists(DATETIME_TABLE) and os.path.exists(ROAD_TABLE)):
        print("creating new context information tables...")
        save_tables(args.save_path)
    dt_table = pd.read_pickle(DATETIME_TABLE)
    rd_table = pd.read_pickle(ROAD_TABLE)
    df = postprocess(df, dt_table, rd_table)
    print("added context information")

    df.to_pickle(args.save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "save_path", help="path to save preprocessed dataframe"
    )
    parser.add_argument("-s", "--start", help="starting date of the period")
    parser.add_argument("-e", "--end", help="ending date of the period")
    args = parser.parse_args()

    main(args)
