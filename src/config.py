# data
DATASET_DIR = "./datasets"
TABLES_DIR = f"{DATASET_DIR}/tables"
MINI_DIR = f"{DATASET_DIR}/mini"

# log
LOG_DIR = "./logs"

# model
MODEL_DIR = "./models"

# features
TIME_COL = ["datetime_id"]
ROAD_COL = ["section_id"]
SEARCH_COL = ["search_1h", "search_unspec_1d"]
TRAFFIC_COL = ["allCars"]
FEATURE_COL = TIME_COL + ROAD_COL + SEARCH_COL + TRAFFIC_COL
TARGET_COL = "allCars"

DT_TABLE_SIZE = 288
SEC_TABLE_SIZE = 63

# parameters
RANDOM_SEED = 42
BATCH_SIZE = 256
TIME_STEP = 168
PREDICTION_HORIZON = 24
SPACE_WINDOW = (-2, 2)

# column num of static features (time id & road id)
TIME_COL_INDEX = 0
ROAD_COL_INDEX = 1
STATIC_COL = [TIME_COL_INDEX, ROAD_COL_INDEX]
SEARCH_COL_INDEX = [-3, -2]
TRAFFIC_COL_INDEX = [-1]
