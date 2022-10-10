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
DF_ALL = './datasets_1h/kannetsu_210402-220531.pkl'
DF_ALL_TRAIN = './datasets_1h/kannetsu_210402-220228.pkl'
DF_ALL_TEST = './datasets_1h/kannetsu_220301-220531.pkl'
DF_MINI = './datasets_1h/kannetsu_210402-210531.pkl'
DF_MINI_TRAIN = './datasets_1h/kannetsu_210402-210519.pkl'
DF_MINI_TEST = './datasets_1h/kannetsu_210520-210531.pkl'

X_TRAIN = './datasets_1h/features_train_norm.pkl'
X_TEST = './datasets_1h/features_test_norm.pkl'
Y_TRAIN = './datasets_1h/labels_train.pkl'
Y_TEST = './datasets_1h/labels_test.pkl'

# CONTEXT TABLES
DATETIME_TABLE = './datasets/tables/datetime_table.pkl'
ROAD_TABLE = './datasets/tables/road_table.pkl'

# standard scaler object
SCALER_PATH = './datasets/training_scaler.pkl'
