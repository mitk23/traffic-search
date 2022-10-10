import itertools

import pandas as pd

from config.storage import DATETIME_TABLE, ROAD_TABLE


def create_datetime_table():
    """
    日時情報のマッピングテーブルを作成
    """
    hours = range(24)
    dayofweeks = range(1, 7 + 1)
    is_holidays = (0, 1)

    dt_table = pd.DataFrame(
        itertools.product(hours, dayofweeks, is_holidays),
        columns=["hour", "dayofweek", "is_holiday"],
        dtype="category",
    )
    dt_table = dt_table.query(
        "dayofweek not in (6, 7) | is_holiday != 0"
    ).reset_index(drop=True)
    dt_table = (
        dt_table.assign(index=dt_table.index)
        .set_index(["hour", "dayofweek", "is_holiday"])
        .astype("category")
    )

    return dt_table


def create_road_table(df_pkl_path):
    """
    df_pkl_pathのデータセットから道路情報を読み出してマッピングテーブルを作成

    Parameters
    ----------
    df_pkl_path: str
    """
    df_mini = pd.read_pickle(df_pkl_path)

    fields = ["start_name", "end_name", "direction", "KP"]

    rd_table = df_mini.loc[:, fields].drop_duplicates()
    # 区間順にソート（上りだったらKP降順, 下りだったらKP昇順）
    sort_f = lambda g: g.sort_values("KP", ascending=(g.name == 1))
    rd_table = (
        rd_table.groupby("direction").apply(sort_f).reset_index(drop=True)
    )

    return rd_table


def save_tables(df_pkl_path, dt_path=DATETIME_TABLE, rd_path=ROAD_TABLE):
    dt_table = create_datetime_table()
    dt_table.to_pickle(dt_path)
    print(f"saved datetime table to {dt_path}")

    rd_table = create_road_table(df_pkl_path)
    rd_table.to_pickle(rd_path)
    print(f"saved datetime table to {rd_path} based on {df_pkl_path}")


def main(args):
    dt_path = args.datetime if args.datetime else DATETIME_TABLE
    rd_path = args.road if args.road else ROAD_TABLE
    save_tables(args.df_pkl_path, dt_path=dt_path, rd_path=rd_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "df_pkl_path",
        help="path of pickled traffic data referenced when creating road table",
    )
    parser.add_argument("-d", "--datetime", help="path to save datetime table")
    parser.add_argument("-r", "--road", help="path to save road table")
    args = parser.parse_args()

    main(args)
