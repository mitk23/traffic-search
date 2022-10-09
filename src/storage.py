'''
データの格納場所を宣言
'''
TARGET_ROAD = "kannetsu"

# data directory
PROCESSED_DATA_DIR = "../Input_processed_data"
# traffic data directory
TRAFFIC_DIR = f"{PROCESSED_DATA_DIR}/traffic"

# E17 data at 1-hour interval
TRAFFIC_CSV = f"{TRAFFIC_DIR}/{TARGET_ROAD}_20220621all-merged_filled_1h.csv"
# column types of TRAFFIC_CSV
COL_TYPES = {
    "start_code": str,
    "end_code": str,
    "road_code": str,
    "jam_type": str,
}

# PICKLE DATA
PKL_ALL = './datasets_1h/kannetsu_210402-220531.pkl'
PKL_MINI = './datasets_1h/kannetsu_210402-210531.pkl'

# CONTEXT TABLES
DATETIME_TABLE = './datasets/tables/datetime_table.pkl'
ROAD_TABLE = './datasets/tables/road_table.pkl'
